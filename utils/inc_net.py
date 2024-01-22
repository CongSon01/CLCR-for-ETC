import copy
import logging
import torch
from torch import _nnpack_available

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained)
    
    # MEMO benchmark backbone
    elif name == 'memo_resnet18':
        _basenet, _adaptive_net = get_memo_resnet18()
        return _basenet, _adaptive_net
    elif name == 'memo_resnet32':
        _basenet, _adaptive_net = get_memo_resnet32()
        return _basenet, _adaptive_net
    
    # AUC
    ## cifar
    elif name == 'conv2':
        return conv2_cifar()
    elif name == 'resnet14_cifar':
        return resnet14_cifar()
    elif name == 'resnet20_cifar':
        return resnet20_cifar()
    elif name == 'resnet26_cifar':
        return resnet26_cifar()
    
    elif name == 'memo_conv2':
        g_blocks, s_blocks = memo_conv2_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet14_cifar':
        g_blocks, s_blocks = memo_resnet14_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet20_cifar':
        g_blocks, s_blocks = memo_resnet20_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet26_cifar':
        g_blocks, s_blocks = memo_resnet26_cifar() # generalized/specialized
        return g_blocks, s_blocks
    
    ## imagenet
    elif name == 'conv4':
        return conv4_imagenet()
    elif name == 'resnet10_imagenet':
        return resnet10_imagenet()
    elif name == 'resnet26_imagenet':
        return resnet26_imagenet()
    elif name == 'resnet34_imagenet':
        return resnet34_imagenet()
    elif name == 'resnet50_imagenet':
        return resnet50_imagenet()
    
    elif name == 'memo_conv4':
        g_blcoks, s_blocks = memo_conv4_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet10_imagenet':
        g_blcoks, s_blocks = memo_resnet10_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet26_imagenet':
        g_blcoks, s_blocks = memo_resnet26_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet34_imagenet':
        g_blocks, s_blocks = memo_resnet34_imagenet()
        return g_blocks, s_blocks
    elif name == 'memo_resnet50_imagenet':
        g_blcoks, s_blocks = memo_resnet50_imagenet()
        return g_blcoks, s_blocks
    else:
        raise NotImplementedError("Unknown type {}".format(convnet_type))
    
class FOSTERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.convnet_type))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc