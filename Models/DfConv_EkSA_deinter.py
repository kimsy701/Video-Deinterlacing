''' network architecture for DfConv_EkSA '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.archs.arch_util as arch_util
import cv2


from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from torch.nn.modules.utils import _pair
class DCN(ModulatedDeformConv2d):
    def __init__(self, *args, extra_offset_mask=False, **kwargs):
        super(DCN, self).__init__(*args, **kwargs)

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups,
                                     self.deform_groups)
                                     
class kNN_Attention_efficient(nn.Module):
    def __init__(self,dim):
        super(kNN_Attention_efficient,self).__init__()
        
        num_heads=1
        qkv_bias=False
        qk_scale=None
        attn_drop=0.
        proj_drop=0.
        topk = 50
        
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5
        self.topk=topk

        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

    def forward(self,x):
        B, C, H, W=x.shape
        N = H*W
        #B,N,C=x.shape
        qkv=self.qkv(x.reshape(B,N,C)).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2] #B,H,N,C
        attn=(v.transpose(-2,-1)@k)*self.scale #B,H,N,N
        # the core code block
        mask=torch.zeros(B,self.num_heads,C,C,device=x.device,requires_grad=False)
        index=torch.topk(attn,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.)
        attn=torch.where(mask>0,attn,torch.full_like(attn,float('-inf')))
        # end of the core code block
        attn=torch.softmax(attn,dim=-1)
        attn=self.attn_drop(attn)
        out=(attn@q.transpose(-2,-1)).reshape(B,N,C)
        out=self.proj(out)
        out=self.proj_drop(out)

        return out.reshape(B,C,H,W) + x


class align_net(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(align_net, self).__init__()

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(ResidualBlock_noBN_f, 5)
        self.relu = nn.ReLU(inplace=True)

        self.dconv_1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.offset_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dconv_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.dconv_3 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dconv_4 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.dconv_5 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dconv_6 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.dconv_7 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        self.offset_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deform_groups=groups, extra_offset_mask=True)
        

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            supp = x[:, i, :, :, :].contiguous()

            # feature trans
            offset1 = self.dconv_1([supp, ref])
            offset1 = self.offset_conv1(offset1)
            fea0 = self.dconv_2([supp, self.relu(offset1)])
            
            offset2 = self.dconv_3([fea0, ref])
            offset2 = self.offset_conv2(offset2)
            fea1 = self.dconv_4([fea0, self.relu(offset2)])
            
            offset3 = self.dconv_5([fea1, ref])
            offset3 = self.offset_conv3(offset3)
            fea2 = self.dconv_6([fea1, self.relu(offset3)])
            
            offset4 = self.dconv_7([fea2, ref])
            offset4 = self.offset_conv4(offset4)
            fea3 = self.dconv([fea2, self.relu(offset4)])
            
            y.append(self.relu(fea3))

        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # extract features
        out = self.relu(self.conv_first(x.view(-1, ch, w, h)))
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        # align supporting frames

        return self.align(out)  # motion alignments

class DfConv_EkSA(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=7, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(DfConv_EkSA, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        
        self.dc_align = align_net(nf=nf, groups=groups)

        #knn_attention_efficient
        self.knnattn_conv_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.knnattn_conv_2 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.knn_attn = kNN_Attention_efficient(nf)
        
        if not self.w_TSA:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk1 = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        # self.recon_trunk2 = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs) #안쓰이니까 주석처리? (실험)

        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x0):
        evenodd=x0[:, -1, :, :, :].contiguous()
        # print("evenodd shape", evenodd.shape) #torch.Size([8, 3, 128, 448]) #각 배치의 가장 마지막 이미지
        x = x0[:, 0:-1, :, :, :].contiguous() #하나 자름 
        # print("x shape",x.shape) #torch.Size([8, 5, 3, 128, 448]) 
        # cv2.imwrite(f'/home/kimsy701/Video-Deinterlacing/x_0_2.png', (x[0][2]*255).clip(0,255).int().cpu().permute(1,2,0).numpy()) #중간꺼 print
        #cv2.imwrite('/home/kimsy701/Video-Deinterlacing/img_l.png',(img_l[0] * 255).clip(0, 255).astype(np.uint8))
        # print("x avg", torch.mean(x)) #0.4000  항상 0.4 
        # print("x[0][2] avg", torch.mean(x[0][2])) #0..

        B, N, C, H, W = x.size()  # N video frames
        eo = []
        for i in range(B):
            eo.append(int('{:.0f}'.format(evenodd[i,0,0,0])))

        x_center = x[:, self.center, :, :, :].contiguous()
        # print("x_center shape", x_center.shape) #torch.Size([8, 3, 128, 448])
        # print("x_center avg", torch.mean(x_center)) #0.  ??
        # print("x_center avg *255", torch.mean(x_center*255)) #0.???
        # cv2.imwrite(f'/home/kimsy701/Video-Deinterlacing/x_center.png', (x_center[0]*255).clip(0,255).int().cpu().permute(1,2,0).numpy())

        #### extract LR features
        aligned_fea = self.dc_align(x)
        # print("aligned_fea len", len(aligned_fea)) #N?  #5 # list
        # print("aligned_fea[0] len",len(aligned_fea[0])) #8
        # print("aligned_fea[0][0] shape", aligned_fea[0][0].shape) #torch.Size([64, 128, 448])
        # cv2.imwrite(f'/home/kimsy701/Video-Deinterlacing/aligned_fea.png', (aligned_fea[2]*255).int().cpu().permute(1,2,0).numpy()) #중간꺼 print


        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        
        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W) 
            # print("aligned_fea shape here", aligned_fea.shape) #torch.Size([8, 320, 128, 448])  #torch.Size([8, 64x5, 128, 448]) 
        
        fea = self.tsa_fusion(aligned_fea)
        # print("fea shape", fea.shape) #torch.Size([8, 64, 128, 448])

        #knn attention
        xf  = self.lrelu(self.knnattn_conv_1(x.view(-1, C, H, W)))
        xf = self.knn_attn(self.knnattn_conv_2(xf.view(B, -1, H, W)))
        
        fea = fea + xf
        # print("xf shape", xf.shape) #torch.Size([8, 64, 128, 448])
        # print("final fea shape", fea.shape) #torch.Size([8, 64, 128, 448])
        
        out = []
        for i in range(B):
            fea0 = fea[i,:,:,:].unsqueeze(0) #torch.Size([1, 64, 128, 448])
            if eo[i]: #when evenodd mode
               out.append(self.recon_trunk1(fea0).squeeze())
            #    print("self.recon_trunk1(fea0) shape", self.recon_trunk1(fea0).shape) #torch.Size([1, 64, 128, 448])
            else: # when not ovenodd mode
            #    out.append(self.recon_trunk2(fea0).squeeze())
                out.append(self.recon_trunk1(fea0).squeeze())
                del fea0
        out = torch.stack(out, dim=0)
        out = self.lrelu(self.HRconv(out))
        # print("self.HRconv(out) shape",self.HRconv(out).shape) #torch.Size([8, 64, 128, 448])
        # print("out lrelu shape",out.shape ) #torch.Size([8, 64, 128, 448])
        out = self.conv_last(out)
        # print("out last conv shape",out.shape) #torch.Size([8, 3, 128, 448])

        base = torch.zeros(B, C, 2*H, W).cuda()
        for i in range(B):
            base[i, :,   eo[i]:2*H:2, :W] = x_center[i, :, :, :]
            base[i, :, 1-eo[i]:2*H:2, :W] = out[i, :, :, :]
        del x_center,out
        return base
