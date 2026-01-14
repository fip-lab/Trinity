import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from models.bert import BertTextEncoder
from models.mamba_change import MambaConfig, MambaBlock
from models.module_layer import Transformer, CrossTransformer, LanguageRouterMoeTransformer, GatedFusion_mlp, AV_Temporal_Attn, Attention, FeedForward, PreNormAttention, PreNormForward, GatedFusion_mlp_eight



class TQFormer_Attention(nn.Module):  
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, attn_mask=None):
        b, nq, _ = q.shape
        _, nk, _ = k.shape
        h = self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            (q, k, v)
        )

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # print("dots shape:", dots.shape, dots) # (B, h, K, L)

        if attn_mask is not None:
            # attn_mask: (1, 1, K, L) or (B, 1, K, L)
            dots = dots + attn_mask

        # print("dots after mask shape:", dots.shape, dots)

        attn = self.attend(dots)
        # print("attn shape:", attn.shape, attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class PreNormCrossAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, attn_mask=None):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v, attn_mask=attn_mask)



class TimeLocalQFormerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.cross_attn = PreNormCrossAttention(
            dim,
            TQFormer_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        )

        self.ff = PreNormForward(
            dim,
            FeedForward(dim, mlp_dim, dropout=dropout)
        )

    def forward(self, q, x, attn_mask):
        # q: (B, K, C)
        # x: (B, L, C)
        q = self.cross_attn(q, x, x, attn_mask=attn_mask) + q
        q = self.ff(q) + q
        return q




class TimeLocalQFormerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeLocalQFormerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, q, x, attn_mask):
        for layer in self.layers:
            q = layer(q, x, attn_mask)
        return q



def build_local_attn_mask(L, K, device):
    mask = torch.full((K, L), float('-inf'), device=device)

    for i in range(K):
        s = int(i * L / K)
        e = int((i + 1) * L / K)
        mask[i, s:e] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)




class TimeLocalQFormerCompressor(nn.Module):
    """
    Local Q-Former for temporal token compression
    (B, L, C) -> (B, K, C)
    """
    def __init__(self, dim, num_queries=50, depth=2, heads=8, dim_head=64, mlp_dim=256, dropout=0.):
        super().__init__()

        self.num_queries = num_queries

        # learnable query tokens (ordered)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, dim)
        )

        self.encoder = TimeLocalQFormerEncoder(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        """
        x: (B, L, C)
        return: (B, K, C)
        """
        b, L, _ = x.shape
        device = x.device

        q = repeat(self.query_tokens, '1 k d -> b k d', b=b)
        attn_mask = build_local_attn_mask(L, self.num_queries, device)
        q = self.encoder(q, x, attn_mask)
        return q





















# Trinity
class Trinity(nn.Module):

    def __init__(self, dataset, fusion_layer_depth=2, num_experts=5, top_k=3, capacity_factor=1.0, dropout=0.0, cls_num=1, token_len=50, vision_feats='', audio_feats='', bert_pretrained='..pretrainedmodel/bert-base-uncased'):
        super(Trinity, self).__init__()

        print("\n\nmodel参数: ", "dataset: ", dataset, "fusion_layer_depth: ", fusion_layer_depth, "num_experts: ", num_experts, "top_k: ", top_k, "capacity_factor: ", capacity_factor, "dropout: ", dropout)

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)
        self.token_len = token_len

        if vision_feats == 'resnet2plus1D':
            v_dim = 512
        elif vision_feats == 'resnet18':
            v_dim = 512
        else:
            # assert False, "vision_feats must be resnet2plus1D or resnet18."
            v_dim = 'msa'

        if audio_feats == 'librosa':
            a_dim = 83
        elif audio_feats == 'vggish':
            a_dim = 128
        else:
            # assert False, "audio_feats must be librosa or vggish."
            a_dim = 'msa'

        print("cls_num: ", cls_num, "     vision_feats: ", vision_feats, v_dim, "  ||  audio_feats: ", audio_feats, a_dim, '\n\n')


        if dataset == 'mosei':
            self.audioFeat_convpool = TimeLocalQFormerCompressor(dim=74, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.audioClue_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.visionFeat_convpool = TimeLocalQFormerCompressor(dim=35, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.visionClue_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.text_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
        elif dataset in ['AVE', 'KS', "UCF51"]:
            self.audioFeat_convpool = TimeLocalQFormerCompressor(dim=a_dim, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.audioClue_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.visionFeat_convpool = TimeLocalQFormerCompressor(dim=v_dim, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.visionClue_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.text_convpool = TimeLocalQFormerCompressor(dim=768, num_queries=token_len, depth=1, heads=8, dim_head=64, mlp_dim=128, dropout=dropout)
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(a_dim, 128)
            self.proj_v0 = nn.Linear(v_dim, 128)
        else:
            assert False, "DatasetName must be mosei or AVE KS UCF51."



        self.proj_a_clue0 = nn.Linear(768, 128)
        self.proj_v_clue0 = nn.Linear(768, 128)

        self.proj_t = Transformer(num_frames=token_len, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.proj_a_feat = Transformer(num_frames=token_len, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.proj_a_clue = Transformer(num_frames=token_len, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.proj_v_feat = Transformer(num_frames=token_len, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.proj_v_clue = Transformer(num_frames=token_len, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.corss_fusion_audio = CrossTransformer(source_num_frames=token_len+1, tgt_num_frames=token_len+1, token_len=None, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.corss_fusion_visual = CrossTransformer(source_num_frames=token_len+1, tgt_num_frames=token_len+1, token_len=None, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.cross_align_a_v = AV_Temporal_Attn(dim=128, inner_dim=256, d_conv=4)
                
        self.trinity_cross_fusion = CrossTransformer(source_num_frames=2*(token_len+1), tgt_num_frames=(token_len+1), token_len=None, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.text_router_moe_trans = LanguageRouterMoeTransformer(num_frames=3*token_len+3, token_len=None, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128, num_experts=num_experts, top_k=top_k, capacity_factor=capacity_factor, dropout=dropout, emb_dropout=dropout)

        self.cls_head = GatedFusion_mlp_eight(dim=128, class_num=cls_num)


    def forward(self, text, audio, visual, audio_clue, visual_clue):

        b = visual.size(0)  # batch size

        text = self.bertmodel(text)
        audio_clue = self.bertmodel(audio_clue)
        visual_clue = self.bertmodel(visual_clue)

        # print("text shape:", text.shape, "  audio shape:", audio.shape, "  visual shape:", visual.shape, "  audio_clue shape:", audio_clue.shape, "  visual_clue shape:", visual_clue.shape)

        text = self.text_convpool(text)
        audio = self.audioFeat_convpool(audio)
        visual = self.visionFeat_convpool(visual)
        audio_clue = self.audioClue_convpool(audio_clue)
        visual_clue = self.visionClue_convpool(visual_clue)

        # print("After convpool - text shape:", text.shape, "  audio shape:", audio.shape, "  visual shape:", visual.shape, "  audio_clue shape:", audio_clue.shape, "  visual_clue shape:", visual_clue.shape)

        text = self.proj_t0(text)
        audio = self.proj_a0(audio)
        visual = self.proj_v0(visual)
        audio_clue = self.proj_a_clue0(audio_clue)
        visual_clue = self.proj_v_clue0(visual_clue)

        # print("After convpool - text shape:", text.shape, "  audio shape:", audio.shape, "  visual shape:", visual.shape, "  audio_clue shape:", audio_clue.shape, "  visual_clue shape:", visual_clue.shape)

        # transformer
        text = self.proj_t(text)

        audio = self.proj_a_feat(audio)
        audio_clue = self.proj_a_clue(audio_clue)
        
        visual = self.proj_v_feat(visual)
        visual_clue = self.proj_v_clue(visual_clue)

        # cross fusion
        audio_fusion = self.corss_fusion_audio(source_x=audio, target_x=audio_clue)
        visual_fusion = self.corss_fusion_visual(source_x=visual, target_x=visual_clue)

        # cross align
        temporal_attn_av = self.cross_align_a_v(audio_fusion, visual_fusion)

        # 三模态时序融合
        tav_temporal_fusion = self.trinity_cross_fusion(source_x=temporal_attn_av, target_x=text)
        # print("tav_temporal_fusion shape:", tav_temporal_fusion.shape)

        cat_feat = torch.cat((text, temporal_attn_av), dim=1)
        moe_trans = self.text_router_moe_trans(text, cat_feat)

        text_extra = moe_trans[:, 0, :]
        audio_extra = moe_trans[:, self.token_len+1, :]
        visual_extra = moe_trans[:, 2*(self.token_len+1), :]
        tav_extra = tav_temporal_fusion[:, 0, :]

        text_poll = torch.max(moe_trans[:, 1:self.token_len+1, :], dim=1)[0]  # (B, 128)
        audio_poll = torch.max(moe_trans[:, self.token_len+2:2*(self.token_len+1), :], dim=1)[0]
        visual_poll = torch.max(moe_trans[:, 2*(self.token_len+1)+1:3*(self.token_len+1), :], dim=1)[0]
        tav_poll = torch.max(tav_temporal_fusion[:, 1:self.token_len+1, :], dim=1)[0]

        logits = self.cls_head(text_extra, audio_extra, visual_extra, tav_extra, text_poll, audio_poll, visual_poll, tav_poll)

        return logits