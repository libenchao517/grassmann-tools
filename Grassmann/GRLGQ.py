################################################################################
# 本文件中是Generalized Relevance Learning Grassmann Quantization
################################################################################
"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main
Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.
"""
################################################################################
# 导入模块
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
LOW_BOUND_LAMBDA = 0.001
################################################################################
# 以下是原作者定义的必要函数和类
def best_train_model(model, best_train_acc: float, train_acc: float):
    model.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    return best_train_acc

def best_test_model(model, best_test_acc: float, target_pred, info):
    model.eval()
    if info['test_accuracy'] > best_test_acc:
        best_test_acc = info['test_accuracy']
        target_pred = info['target_pred']
    return best_test_acc, target_pred

def metrics(y_true: Tensor, y_pred: Tensor, nclasses):
    assert y_true.shape == y_pred.shape, f'their shape is labels: {y_true.shape}, pred:{y_pred.shape}'
    acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    c = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=range(nclasses))
    return acc, c

def get_optimizer(model, lr_protos, lr_rel):
    params_prototypes = []
    params_relevances = []
    for name, param in model.named_parameters():
        if 'xprotos' in name:
            params_prototypes.append(param)
        elif 'rel' in name:
            params_relevances.append(param)
        else:
            print(f"There are some parameter not being prototypes and relevances with name: {name}")
    proto_param_list = [{'params': params_prototypes, 'lr': lr_protos}]
    rel_param_list = [{'params': params_relevances, 'lr': lr_rel}]
    return torch.optim.SGD(proto_param_list), torch.optim.SGD(rel_param_list), params_prototypes, params_relevances

def winner_prototype_indices(ydata: Tensor, yprotos_mat: Tensor, distances: Tensor):
    """
    Find the closest prototypes to a batch of features
    :param ydata: labels of input images, SHAPE: (batch_size,)
    :param yprotos_mat: labels of prototypes, SHAPE: (nclass, number_of_prototypes)
    Note: we can use it for both prototypes with the same or different labels (W^+ and W^-)
    :param distances: distances between images and prototypes, SHAPE: (batch_size, number_of_prototypes)
    :return: a dictionary containing winner prototypes
    """
    assert distances.ndim == 2, (f"There should be a distance matrix of shape (batch_size, number_of_prototypes), but it gets {distances.shape}")
    Y = yprotos_mat[ydata.tolist()]
    distances_sparse = distances * Y
    return torch.stack([torch.argwhere(w).T[0, torch.argmin(w[torch.argwhere(w).T],)] for w in torch.unbind(distances_sparse)], dim=0).T

def winner_prototype_distances(ydata: Tensor, yprotos_matrix: Tensor, yprotos_comp_matrix: Tensor, distances: Tensor):
    """
    find the distance between winners' prototypes and data
    :param ydata: a (nbatch,) array containing labels of data
    :param yprotos_matrix: a (nclass, nprotos) matrix containing non-zero elements in c-th row for prototypes with label 'c'
    :param distantces: (nbatch, nprotos) matrix containing distances between data and prototypes
    :return: D^{+,-} matrices of size (nbatch, nprotos) containing zero on not-winner prototypes
    """
    nbatch, nprotos = distances.shape
    iplus = winner_prototype_indices(ydata, yprotos_matrix, distances)
    iminus = winner_prototype_indices(ydata, yprotos_comp_matrix, distances)
    Dplus = torch.zeros_like(distances)
    Dminus = torch.zeros_like(distances)
    Dplus[torch.arange(nbatch), iplus] = distances[torch.arange(nbatch), iplus]
    Dminus[torch.arange(nbatch), iminus] = distances[torch.arange(nbatch), iminus]
    return Dplus, Dminus, iplus, iminus

def MU_fun(Dplus, Dminus):
    """
    Mu = (D^+ - D^-)/(D^++D^-)
    :param ydata: a (nbatch,) array containing labels of data
    :param yprotos_matrix: a (nclass, nprotos) matrix containing non-zero elements in c-th row for prototypes with label 'c'
    :param distantces: (nbatch, nprotos) matrix containing distances between data and prototypes
    :return: an array of size (nbatch,) containing mu values
    """
    return (Dplus - Dminus).sum(axis=1) / (Dplus + Dminus).sum(axis=1)

def IdentityLoss():
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.Identity()(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def SigmoidLoss(sigma: int=100):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.Sigmoid()(sigma * MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def ReLULoss():
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.ReLU()(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def LeakyReLULoss(negative_slope: float=0.01):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.LeakyReLU(negative_slope)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def ELULoss(alpha: float=1):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.ELU(alpha)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def RReLULoss(lower=0.125, upper=0.3333333333333333):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (torch.nn.RReLU(lower, upper)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f

def get_loss_function(cost_fun, sigma):
    if cost_fun == 'sigmoid':
        sigma = sigma or 100
        return SigmoidLoss(sigma)
    elif cost_fun == 'relu':
        return ReLULoss()
    elif cost_fun == 'leaky_relu':
        sigma = sigma or 0.1
        return LeakyReLULoss(sigma)
    elif cost_fun == 'elu':
        sigma = sigma or 1
        return ELULoss(alpha=sigma)
    elif cost_fun == 'rrelu':
        return RReLULoss()
    else:
        return IdentityLoss()

def init_randn(dim_of_data: int, dim_of_subspace: int, labels: Tensor = None, num_of_protos: [int, Tensor] = 1, num_of_classes: int = None, device='cpu'):
    if labels is None:
        classes = torch.arange(num_of_classes)
    else:
        classes = torch.unique(labels)
    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
    else:
        total_num_of_protos = torch.sum(num_of_protos).item()
    nclass = len(classes)
    prototype_shape = (total_num_of_protos, dim_of_data, dim_of_subspace)
    Q, _ = torch.linalg.qr(torch.randn(prototype_shape), mode='reduced')
    xprotos = torch.nn.Parameter(Q)
    yprotos = torch.from_numpy(np.repeat(classes.numpy(), num_of_protos)).to(torch.int32)
    yprotos_mat = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)
    yprotos_mat_comp = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)
    # setting prototypes' labels
    for i, proto in enumerate(xprotos):
        yprotos_mat[yprotos[i], i] = 1
        tmp = list(range(len(classes)))
        tmp.pop(yprotos[i])
        yprotos_mat_comp[tmp, i] = 1
    return xprotos.to(device), yprotos.to(device), yprotos_mat.to(device), yprotos_mat_comp.to(device)

def grassmann_repr(batch_imgs: Tensor, dim_of_subspace: int) -> Tensor:
    """
    :param batch_imgs: a batch of features of size (batch size, num_of_channels, W, H)
    :param dim_of_subspace: the dimensionality of the extracted subspace
    :return: an orthonormal matrix of size (batch size, W*H, dim_of_subspace)
    """
    assert batch_imgs.ndim == 4, f"xs should be of the shape (batch_size, nchannel, w, h), but it is {batch_imgs.shape}"
    bsize, nchannel, w, h = batch_imgs.shape
    xs = batch_imgs.view(bsize, nchannel, w * h)
    # SVD: generate principal directions
    U, S, Vh = torch.linalg.svd(xs, full_matrices=False)
    if U.shape[0] > U.shape[1]:
        return U[:, :, :dim_of_subspace]
    else:
        return Vh.transpose(-1, -2)[:, :, :dim_of_subspace]

def compute_distances_on_grassmann_mdf(xdata: Tensor, xprotos: Tensor, metric_type: str = 'chordal', relevance: Tensor = None):
    """
    Compute the (geodesic or chordal) distances between an input subspace and all prototypes.
    """
    assert xdata.ndim == 3, f"xs should be of the shape (batch_size, W*H, dim_of_subspace), but it is {xdata.shape}"
    if relevance is None:
        relevance = torch.ones((1, xprotos.shape[-1])) / xprotos.shape[-1]
    xdata = xdata.unsqueeze(dim=1)
    U, S, Vh = torch.linalg.svd(torch.transpose(xdata, 2, 3) @ xprotos.to(xdata.dtype), full_matrices=False)
    if metric_type == 'chordal':
        distance = 1 - torch.transpose(relevance @ torch.transpose(S, 1, 2).to(relevance.dtype), 1, 2)
    else:
        distance = torch.transpose(relevance @ torch.transpose(torch.acos(S) ** 2, 1, 2).to(relevance.dtype), 1, 2)
    if torch.isnan(distance).any():
        raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')
    output = {
        'Q': U, # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'Qw': torch.transpose(Vh, 2, 3), # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'canonicalcorrelation': S, # SHAPE: (batch_size, num_of_prototypes, dim_of_subspaces)
        'distance': torch.squeeze(distance, -1)} # SHAPE: (batch_size, num_of_prototypes)}
    return output

def orthogonalize_batch(x_batch: Tensor) -> Tensor:
    Q, _ = torch.linalg.qr(x_batch, mode='reduced')
    return Q

def rotate_data(xs, rotation_matrix, winner_ids, return_rotation_matrix=False):
    assert xs.ndim == 3, f"data should be of shape (batch_size, dim_of_data, dim_of_subspace), but it is of shape {xs.shape}"
    assert winner_ids.shape[1] == 2, f"There should only be two winners W^+- prototypes for each data. But now there are {winner_ids.shape[1]} winners."
    nbatch = xs.shape[0]
    Qwinners = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids]  # shape: (batch_size, 2, d, d)
    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]  # shape: (batch_size, d, d)
    rotated_xs1, rotated_xs2 = torch.bmm(xs, Qwinners1), torch.bmm(xs, Qwinners2)  # shape: (batch_size, D, d)
    if return_rotation_matrix:
        return rotated_xs1, rotated_xs2, Qwinners1, Qwinners2
    return rotated_xs1, rotated_xs2

def rotate_prototypes(xprotos, rotation_matrix, winner_ids):
    assert xprotos.ndim == 3, f"data should be of shape (nprotos, dim_of_data, dim_of_subspace), but it is of shape {xprotos.shape}"
    assert winner_ids.shape[1] == 2, f"There should only be two winners W^+- prototypes for each data. But now there are {winner_ids.shape[1]} winners."
    nbatch, nprotos = rotation_matrix.shape[:2]
    Qwinners = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids]  # shape: (batch_size, 2, d, d)
    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]
    assert Qwinners1.shape[0] == nbatch, f"The size of Qwinner should be (nbatch, ...) but it is {Qwinners1.shape}"
    xprotos_winners = xprotos[winner_ids]
    xprotos1, xprotos2 = xprotos_winners[:, 0], xprotos_winners[:, 1]
    rotated_proto1 = torch.bmm(xprotos1, Qwinners1.to(xprotos1.dtype))
    rotated_proto2 = torch.bmm(xprotos2, Qwinners2.to(xprotos1.dtype))
    return rotated_proto1, rotated_proto2

def train_epoch(model, train_loader: DataLoader, epoch: int, loss, batch_size_train, optimizer_protos: torch.optim.Optimizer, optimizer_rel: torch.optim.Optimizer, device, progress_prefix: str = 'Train Epoch') -> dict:
    model = model.to(device)
    # to store information about the procedure
    train_info = dict()
    total_loss = 0
    total_acc = 0
    # to show the progress-bar
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix + ' %s' % epoch, ncols=0)
    # training process (one epoch)
    for i, (xtrain, ytrain) in enumerate(train_loader):
        # ****** for the first solution
        optimizer_protos.zero_grad()
        optimizer_rel.zero_grad()
        xtrain, ytrain = xtrain.to(device), ytrain.to(device)
        distances, Qw = model.prototype_layer(xtrain)
        cost, iplus, iminus = loss(ytrain, model.prototype_layer.yprotos_mat, model.prototype_layer.yprotos_comp_mat, distances)
        cost.backward()
        ##### First way: using optimizers ##############
        with torch.no_grad():
            winners_ids, _ = torch.stack([iplus, iminus], axis=1).sort(axis=1)
            rotated_proto1, rotated_proto2 = rotate_prototypes(model.prototype_layer.xprotos, Qw, winners_ids)
            model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 0]] = rotated_proto1
            model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 1]] = rotated_proto2
        optimizer_protos.step()
        optimizer_rel.step()
        with torch.no_grad():
            model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 0]] = orthogonalize_batch(model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 0]])
            model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 1]] = orthogonalize_batch(model.prototype_layer.xprotos[winners_ids[torch.arange(batch_size_train), 1]])
            #CHECK
            LOW_BOUND_LAMBDA = 0.0001
            model.prototype_layer.relevances[0, torch.argwhere(model.prototype_layer.relevances < LOW_BOUND_LAMBDA)[:, 1]] = LOW_BOUND_LAMBDA
            model.prototype_layer.relevances[:] = model.prototype_layer.relevances[:] / model.prototype_layer.relevances.sum()
        # compute the accuracy
        yspred = model.prototype_layer.yprotos[distances.argmin(axis=1)]
        acc = torch.sum(torch.eq(yspred, ytrain)).item() / float(len(xtrain))
        train_iter.set_postfix_str(f"Batch [{i + 1}/{len(train_loader)}, Loss: {cost.sum().item(): .3f}, Acc: {acc: .3f}")
        # update the total metrics
        total_acc += acc
        total_loss += torch.sum(cost).item()
    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)
    return train_info

@torch.no_grad()
def eval(model, test_loader: DataLoader, epoch: int, loss, device, progress_prefix: str = 'Eval Epoch') -> dict :
    model = model.to(device)
    # to store information about the procedure
    test_info = dict()
    model.eval()
    # to show the progress-bar
    train_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc=progress_prefix + ' %s' % epoch, ncols=0)
    conf_mat = np.zeros((model._num_classes, model._num_classes), dtype=int)
    target_pred = torch.empty(size=(0,))
    # training process (one epoch)
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)
        # forward pass
        distances, _ = model.prototype_layer(xs)
        # predict labels
        yspred = model.prototype_layer.yprotos[distances.argmin(axis=1)]
        target_pred = torch.concat((target_pred, yspred))
        cost, iplus, iminus = loss(ys, model.prototype_layer.yprotos_mat, model.prototype_layer.yprotos_comp_mat, distances)
        # compute the confusion matrix
        acc, cmat = metrics(ys, yspred, nclasses=model._num_classes)
        conf_mat += cmat
        train_iter.set_postfix_str(f"Batch [{i + 1}/{len(test_loader)}, Loss: {cost.item(): .3f}, Acc: {acc: .3f}")
    test_info['confusion_matrix'] = conf_mat
    test_info['test_accuracy'] = np.diag(conf_mat).sum() / conf_mat.sum()
    test_info['target_pred'] = np.array(target_pred)
    return test_info

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.images = data
        self.labels = target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), torch.tensor(self.labels[index])

class GeodesicPrototypeLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs_subspace, xprotos, relevances):
        # Compute distances between data and prototypes
        output = compute_distances_on_grassmann_mdf(xs_subspace, xprotos, 'geodesic', relevances)

        ctx.save_for_backward(
            xs_subspace, xprotos, relevances,
            output['distance'], output['Q'], output['Qw'], output['canonicalcorrelation'])
        return output['distance'], output['Qw']

    @staticmethod
    def backward(ctx, grad_output, grad_qw):
        nbatch = grad_output.shape[0]
        xs_subspace, xprotos, relevances, distances, Q, Qw, cc = ctx.saved_tensors
        winner_ids = torch.stack([torch.nonzero(gd).T[0] if len(torch.nonzero(gd).T[0]) == 2 else torch.tensor([-1, -2]) for gd in torch.unbind(grad_output)], dim=0)
        if len(torch.argwhere(winner_ids < 0)) > 0:
            s = torch.argwhere((winner_ids > 0)[:, 0]).T[0]
            xs_subspace = xs_subspace[s]
            distances = distances[s]
            Q = Q[s]
            Qw = Qw[s]
            cc = cc[s]
            winner_ids = winner_ids[s]
            nbatch = s.shape[0]
        # **********************************************
        # ********** gradient of prototypes ************
        # **********************************************
        # Rotate data points (based on winner prototypes)
        rotated_xs1, rotated_xs2, Qwinners1, Qwinners2 = rotate_data(xs_subspace, Q, winner_ids, return_rotation_matrix=True)
        dist_grad1 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]]
        dist_grad2 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]]
        # Find
        thetta_winners1 = torch.acos(cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]])
        thetta_winners2 = torch.acos(cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]])
        assert relevances.shape[1] == thetta_winners1.shape[-1], f"They are not equal {relevances.shape[1]}{thetta_winners1.shape[-1]}"
        #CHECK
        diag_rel_x_thetta1 = relevances[0] * thetta_winners1 / torch.sqrt(1 - torch.cos(thetta_winners1)**2)
        diag_rel_x_thetta2 = relevances[0] * thetta_winners2 / torch.sqrt(1 - torch.cos(thetta_winners2)**2)
        # gradient of prototypes
        grad_protos1 = - rotated_xs1 * diag_rel_x_thetta1.unsqueeze(1) * dist_grad1.unsqueeze(-1).unsqueeze(-1)
        grad_protos2 = - rotated_xs2 * diag_rel_x_thetta2.unsqueeze(1) * dist_grad2.unsqueeze(-1).unsqueeze(-1)
        # **********************************************
        # ********** gradient of relevances ************
        # **********************************************
        #CHECK
        grad_rel = - (thetta_winners1**2 * dist_grad1.unsqueeze(-1) + thetta_winners2**2 * dist_grad2.unsqueeze(-1))
        grad_xs = grad_protos = grad_relevances = None
        if ctx.needs_input_grad[0]:  # TODO: set input grad to false
            print("Hey why the input need gradient, please check it!")
        if ctx.needs_input_grad[1]:
            grad_protos = torch.zeros_like(xprotos)
            grad_protos[winner_ids[torch.arange(nbatch), 0]] = grad_protos1.to(grad_protos.dtype)
            grad_protos[winner_ids[torch.arange(nbatch), 1]] = grad_protos2.to(grad_protos.dtype)
        if ctx.needs_input_grad[2]:
            grad_relevances = grad_rel
        return grad_xs, grad_protos, grad_relevances

class PrototypeLayer(torch.nn.Module):
    def __init__(self, num_prototypes, num_classes, dim_of_data, dim_of_subspace, metric_type='geodesic', dtype=torch.float32, device='cpu'):
        super().__init__()
        self.nchannels = dim_of_data
        self.dim_of_subspaces = dim_of_subspace
        # Each prototype is a latent representation of shape (D, d)
        self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(self.nchannels, self.dim_of_subspaces, num_of_protos=num_prototypes, num_of_classes=num_classes, device=device)
        self.metric_type = metric_type
        self.number_of_prototypes = self.yprotos.shape[0]
        self.relevances = torch.nn.Parameter(torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device) / self.xprotos.shape[-1])

    def forward(self, xs_subspace):
        return GeodesicPrototypeLayer.apply(xs_subspace, self.xprotos, self.relevances)
################################################################################
# 以下是原作者定义的GRLGQ类
class GRLGQ_Model(torch.nn.Module):
    def __init__(self, img_size: int, num_classes: int, cost_fun, num_of_protos, dim_of_subspace, device='cpu'):
        super().__init__()
        assert num_classes > 0
        self._num_classes = num_classes
        self._act_fun = cost_fun
        self._metric_type = 'geodesic'
        self._img_size = img_size
        self._num_prototypes = num_of_protos
        self.prototype_layer = PrototypeLayer(
            num_prototypes=self._num_prototypes,
            num_classes=self._num_classes,
            dim_of_data=self._img_size,
            dim_of_subspace=dim_of_subspace,
            metric_type=self._metric_type,
            device=device)

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.xprotos.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.xprotos.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self, xs: Tensor):
        """
        Compute the distance between features (from Neural Net) and the prototypes
        xs: a batch of subspaces
        """
        # SVD decomposition: representation of net features as a point on the grassmann manifold
        xs_subspaces = grassmann_repr(xs, self.dim_of_subspaces)
        distance, Qw = self.prototype_layer(xs_subspaces) # SHAPE:(batch_size, num_prototypes, D: dim_of_data, d: dim_of_subspace)
        return distance, Qw
################################################################################
class GRLGQ_Run:
    """
    以下是我们封装的调用GRLGQ的类
    [1] Mohammadi M, Babai M, Wilkinson M H F.
    Generalized relevance learning grassmann quantization[J].
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
    [2] https://github.com/mohammadimathstar/GRLGQ
    """
    def __init__(
            self,
            batch_size_train=1,
            batch_size_test=10,
            nepochs=100,
            cost_fun='identity',
            sigma=None,
            metric_type="geodesic",
            num_of_protos=1,
            dim_of_subspace=5,
            lr_protos=0.05,
            lr_rel=0.0001,
            milestones=[],
            gamma=0.5,
            disable_cuda='store_true',
            device = torch.device("cpu")):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.nepochs = nepochs
        self.cost_fun = cost_fun
        self.sigma = sigma
        self.metric_type = metric_type
        self.num_of_protos = num_of_protos
        self.dim_of_subspace = dim_of_subspace
        self.lr_protos = lr_protos
        self.lr_rel = lr_rel
        self.milestones = milestones
        self.gamma = gamma
        self.disable_cuda = disable_cuda
        self.device = device

    def fit(self, data_train, data_test, target_train, target_test):
        device = torch.device(self.device)
        dataset_train = DataSet(data=data_train, target=target_train)
        dataset_test = DataSet(data=data_test, target=target_test)
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size_train, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=self.batch_size_test, shuffle=False)
        model = GRLGQ_Model(
            img_size=data_train.shape[1],
            num_classes=len(np.unique(target_train)),
            cost_fun=self.cost_fun,
            num_of_protos=self.num_of_protos,
            dim_of_subspace=self.dim_of_subspace)
        model = model.to(device=self.device)
        loss = get_loss_function(self.cost_fun, self.sigma)

        optimizer_protos, optimizer_rel, _, _ = get_optimizer(model, self.lr_protos, self.lr_rel)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_protos, milestones=self.milestones, gamma=self.gamma)

        best_train_acc = 0.
        best_test_acc = 0.
        target_pred = []

        for epoch in range(1, self.nepochs + 1):
            train_info = train_epoch(model, train_loader, epoch, loss, self.batch_size_train, optimizer_protos, optimizer_rel, device, progress_prefix='Train Epoch')

            # TODO: complete the following
            best_train_acc = best_train_model(model, best_train_acc, train_info['train_accuracy'])

            eval_info = eval(model, test_loader, epoch, loss, device)
            best_test_acc, target_pred = best_test_model(model, best_test_acc, target_pred, eval_info)
            # update parameters
            scheduler.step()

        return target_pred, best_test_acc

    def fit_transform(self, data_train, data_test, target_train, target_test):
        self.t_pred, self.accuracy = self.fit(data_train, data_test, target_train, target_test)
