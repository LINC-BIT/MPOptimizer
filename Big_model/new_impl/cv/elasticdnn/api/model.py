from typing import List
import torch
from methods.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util

from utils.dl.common.model import LayerActivation


class ElasticDNN_OfflineFMModel(BaseModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    @abstractmethod
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        pass
    
    @abstractmethod
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_feature_hook(self) -> LayerActivation:
        pass
        
    @abstractmethod
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        pass
    
    @abstractmethod
    def get_lora_util(self) -> FMLoRA_Util:
        pass
    
    @abstractmethod
    def get_task_head_params(self):
        pass
    

class ElasticDNN_OfflineClsFMModel(ElasticDNN_OfflineFMModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.infer(x)
                #pred = F.softmax(output.logits, dim=1).argmax(dim=1)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                #correct = torch.eq(torch.argmax(output.logits,dim = 1), y).sum().item()
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        #print(self.models_dict['main'](x))
        return self.models_dict['main'](x)


import numpy as np
class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class ElasticDNN_OfflineSegFMModel(ElasticDNN_OfflineFMModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        device = self.device
        self.to_eval_mode()
        metrics = StreamSegMetrics(self.num_classes)
        metrics.reset()
        import tqdm
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x, y) in pbar:
                x, y = x.to(device, dtype=x.dtype, non_blocking=True, copy=False), \
                    y.to(device, dtype=y.dtype, non_blocking=True, copy=False)
                #output = self.infer(x)
                output = self.infer(x)
                pred = output.detach().max(dim=1)[1].cpu().numpy()
                metrics.update((y + 0).cpu().numpy(), pred)
                
                res = metrics.get_results()
                pbar.set_description(f'cur batch mIoU: {res["Mean IoU"]:.4f}')
                
        res = metrics.get_results()
        return res['Mean IoU']
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    
class ElasticDNN_OfflineDetFMModel(ElasticDNN_OfflineFMModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        # print('DeeplabV3: start test acc')
        _d = test_loader.dataset
        from data import build_dataloader
        if _d.__class__.__name__ == 'MergedDataset':
            # print('\neval on merged datasets')
            datasets = _d.datasets
            test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None) for d in datasets]
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            # print(accs)
            return sum(accs) / len(accs)
        
        # print('dataset len', len(test_loader.dataset))

        model = self.models_dict['main']
        device = self.device
        model.eval()

        # print('# classes', model.num_classes)
        
        model = model.to(device)
        from dnns.yolov3.coco_evaluator import COCOEvaluator
        from utils.common.others import HiddenPrints
        with torch.no_grad():
            with HiddenPrints():
                evaluator = COCOEvaluator(
                    dataloader=test_loader,
                    img_size=(224, 224),
                    confthre=0.01,
                    nmsthre=0.65,
                    num_classes=self.num_classes,
                    testdev=False
                )
                res = evaluator.evaluate(model, False, False)
                map50 = res[1]
            # print('eval info', res[-1])
        return map50
    
    def infer(self, x, *args, **kwargs):
        if len(args) > 0:
            print(args, len(args))
            return self.models_dict['main'](x, *args) # forward(x, label)
        return self.models_dict['main'](x)
    

class ElasticDNN_OfflineSenClsFMModel(ElasticDNN_OfflineFMModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    
from accelerate.utils.operations import pad_across_processes
    
    
class ElasticDNN_OfflineTrFMModel(ElasticDNN_OfflineFMModel):
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        # TODO: BLEU
        from sacrebleu import corpus_bleu
        
        acc = 0
        num_batches = 0
        
        self.to_eval_mode()
        
        from data.datasets.sentiment_classification.global_bert_tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        
        def _decode(o):
            # https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/finetune.py#L133
            o = tokenizer.batch_decode(o, skip_special_tokens=True)
            return [oi.strip() for oi in o]
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                label = y.to(self.device)
                
                # generated_tokens = self.infer(x, generate=True)

                generated_tokens = self.infer(x).logits.argmax(-1)
                
                # pad tokens
                generated_tokens = pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                # pad label
                label = pad_across_processes(
                    label, dim=1, pad_index=tokenizer.pad_token_id
                )
                label = label.cpu().numpy()
                label = np.where(label != -100, label, tokenizer.pad_token_id)
                
                decoded_output = _decode(generated_tokens)
                decoded_y = _decode(y)
                
                decoded_y = [decoded_y]
                
                if batch_index == 0:
                    print(decoded_y, decoded_output)
                
                bleu = corpus_bleu(decoded_output, decoded_y).score
                pbar.set_description(f'cur_batch_bleu: {bleu:.4f}')
                
                acc += bleu
                num_batches += 1

        acc /= num_batches
        return acc
    
    def infer(self, x, *args, **kwargs):
        if 'token_type_ids' in x.keys():
            del x['token_type_ids']
        
        if 'generate' in kwargs:
            return self.models_dict['main'].generate(
                x['input_ids'], 
                attention_mask=x["attention_mask"],
                max_length=512
            )
        
        return self.models_dict['main'](**x)
    

from nltk.metrics import accuracy as nltk_acc

class ElasticDNN_OfflineTokenClsFMModel(ElasticDNN_OfflineFMModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # print(x)
                y = y.to(self.device)
                output = self.infer(x)
                
                # torch.Size([16, 512, 43]) torch.Size([16, 512])
                
                for oi, yi, xi in zip(output, y, x['input_ids']):
                    # oi: 512, 43; yi: 512
                    seq_len = xi.nonzero().size(0)
                    
                    # print(output.size(), y.size())
                    
                    pred = F.softmax(oi, dim=-1).argmax(dim=-1)
                    correct = torch.eq(pred[1: seq_len], yi[1: seq_len]).sum().item()
                    
                    # print(output.size(), y.size())
                    
                    acc += correct
                    sample_num += seq_len
                
                    pbar.set_description(f'seq_len: {seq_len}, cur_seq_acc: {(correct / seq_len):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
class ElasticDNN_OfflineMMClsFMModel(ElasticDNN_OfflineFMModel):
    # def __init__(self, name: str, models_dict_path: str, device: str, class_to_label_idx_map):
    #     super().__init__(name, models_dict_path, device)
    #     self.class_to_label_idx_map = class_to_label_idx_map
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        batch_size = 1
    
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                if batch_index * batch_size > 2000:
                    break
                
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                
                # print(x)
                
                raw_texts = x['texts'][:]
                x['texts'] = list(set(x['texts']))
                
                # print(x['texts'])
                
                batch_size = len(y)
                
                x['for_training'] = False
                
                output = self.infer(x)

                output = output.logits_per_image
                
                # print(output.size())
                # exit()
                
                # y = torch.arange(len(y), device=self.device)
                y = torch.LongTensor([x['texts'].index(rt) for rt in raw_texts]).to(self.device)
                # print(y)
                
                # exit()
                
                # print(output.size(), y.size())
                
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')
                
        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        x['for_training'] = self.models_dict['main'].training
        return self.models_dict['main'](**x)
    


class VQAScore:
    def __init__(self):
        # self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.score = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total



class ElasticDNN_OfflineVQAFMModel(ElasticDNN_OfflineFMModel):
    # def __init__(self, name: str, models_dict_path: str, device: str, class_to_label_idx_map):
    #     super().__init__(name, models_dict_path, device)
    #     self.class_to_label_idx_map = class_to_label_idx_map
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        vqa_score = VQAScore()
        
        self.to_eval_mode()
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained")
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y, t) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                if isinstance(y,dict):
                    for k, v in y.items():
                        y[k] = v.to(self.device)
                else:
                    y = y.to(self.device)
                
                output = self.models_dict['main'].generate(**x)
                total = 0
                idx = 0
                for i in output:
                    val = processor.decode(i, skip_special_tokens=True)
                    text = t[idx]
                    if val == text:
                        total += 1
                    idx += 1
                    
                #vqa_score.update(output, y.labels)
                acc = total / (idx+1)
                #pbar.set_description(f'cur_batch_total: {len(y['label'])}, cur_batch_acc: {vqa_score.compute():.4f}')
                pbar.set_description(f'cur_batch_total: {len(y["labels"])}, cur_batch_acc: {acc:.4f}')
        #return vqa_score.compute()
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
class ElasticDNN_OfflineMDModel(BaseModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    @abstractmethod
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_feature_hook(self) -> LayerActivation:
        pass
    
    @abstractmethod
    def get_distill_loss(self, student_output, teacher_output):
        pass
    
    @abstractmethod
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        pass
    

class ElasticDNN_OfflineClsMDModel(ElasticDNN_OfflineMDModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.infer(x)
                #pred = F.softmax(output.logits, dim=1).argmax(dim=1)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
        
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    

class ElasticDNN_OfflineSegMDModel(ElasticDNN_OfflineMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        device = self.device
        self.to_eval_mode()
        metrics = StreamSegMetrics(self.num_classes)
        metrics.reset()
        import tqdm
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x, y) in pbar:
                x, y = x.to(device, dtype=x.dtype, non_blocking=True, copy=False), \
                    y.to(device, dtype=y.dtype, non_blocking=True, copy=False)
                output = self.infer(x)
                pred = output.detach().max(dim=1)[1].cpu().numpy()
                metrics.update((y + 0).cpu().numpy(), pred)
                
                res = metrics.get_results()
                pbar.set_description(f'cur batch mIoU: {res["Mean IoU"]:.4f}')
                
        res = metrics.get_results()
        return res['Mean IoU']
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    
class ElasticDNN_OfflineDetMDModel(ElasticDNN_OfflineMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        # print('DeeplabV3: start test acc')
        _d = test_loader.dataset
        from data import build_dataloader
        if _d.__class__.__name__ == 'MergedDataset':
            # print('\neval on merged datasets')
            datasets = _d.datasets
            test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None) for d in datasets]
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            # print(accs)
            return sum(accs) / len(accs)
        
        # print('dataset len', len(test_loader.dataset))

        model = self.models_dict['main']
        device = self.device
        model.eval()

        # print('# classes', model.num_classes)
        
        model = model.to(device)
        from dnns.yolov3.coco_evaluator import COCOEvaluator
        from utils.common.others import HiddenPrints
        with torch.no_grad():
            with HiddenPrints():
                evaluator = COCOEvaluator(
                    dataloader=test_loader,
                    img_size=(224, 224),
                    confthre=0.01,
                    nmsthre=0.65,
                    num_classes=self.num_classes,
                    testdev=False
                )
                res = evaluator.evaluate(model, False, False)
                map50 = res[1]
            # print('eval info', res[-1])
        return map50
    
    def infer(self, x, *args, **kwargs):
        if len(args) > 0:
            return self.models_dict['main'](x, *args) # forward(x, label)
        return self.models_dict['main'](x)
    
    
class ElasticDNN_OfflineSenClsMDModel(ElasticDNN_OfflineMDModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                
                print(pred, y)
                
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
        
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
class ElasticDNN_OfflineTrMDModel(ElasticDNN_OfflineMDModel):
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        # TODO: BLEU
        from sacrebleu import corpus_bleu
        
        acc = 0
        num_batches = 0
        
        self.to_eval_mode()
        
        from data.datasets.sentiment_classification.global_bert_tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        
        def _decode(o):
            # https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/finetune.py#L133
            o = tokenizer.batch_decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return [oi.strip().replace(' ', '') for oi in o]
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                
                output = self.infer(x)
                decoded_output = _decode(output.argmax(-1))
                decoded_y = _decode(y)
                
                decoded_y = [decoded_y]
                
                print(x, decoded_y, decoded_output, output.argmax(-1))
                
                bleu = corpus_bleu(decoded_output, decoded_y).score
                pbar.set_description(f'cur_batch_bleu: {bleu:.4f}')
                
                acc += bleu
                num_batches += 1

        acc /= num_batches
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
class ElasticDNN_OfflineTokenClsMDModel(ElasticDNN_OfflineMDModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # print(x)
                y = y.to(self.device)
                output = self.infer(x)
                
                # torch.Size([16, 512, 43]) torch.Size([16, 512])
                
                for oi, yi, xi in zip(output, y, x['input_ids']):
                    # oi: 512, 43; yi: 512
                    seq_len = xi.nonzero().size(0)
                    
                    # print(output.size(), y.size())
                    
                    pred = F.softmax(oi, dim=-1).argmax(dim=-1)
                    correct = torch.eq(pred[1: seq_len], yi[1: seq_len]).sum().item()
                    
                    # print(output.size(), y.size())
                    
                    acc += correct
                    sample_num += seq_len
                
                    pbar.set_description(f'seq_len: {seq_len}, cur_seq_acc: {(correct / seq_len):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
class ElasticDNN_OfflineMMClsMDModel(ElasticDNN_OfflineMDModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        batch_size = 1
    
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                if batch_index * batch_size > 2000:
                    break
                
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                
                # print(x)
                
                raw_texts = x['texts'][:]
                x['texts'] = list(set(x['texts']))
                
                # print(x['texts'])
                
                batch_size = len(y)
                
                x['for_training'] = False
                
                output = self.infer(x)

                output = output.logits_per_image
                
                # print(output.size())
                # exit()
                
                # y = torch.arange(len(y), device=self.device)
                y = torch.LongTensor([x['texts'].index(rt) for rt in raw_texts]).to(self.device)
                # print(y)
                
                # exit()
                
                # print(output.size(), y.size())
                
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')
                
        acc /= sample_num
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    
# class ElasticDNN_OfflineVQAMDModel(ElasticDNN_OfflineMDModel):
#     # def __init__(self, name: str, models_dict_path: str, device: str, class_to_label_idx_map):
#     #     super().__init__(name, models_dict_path, device)
#     #     self.class_to_label_idx_map = class_to_label_idx_map
        
#     def get_accuracy(self, test_loader, *args, **kwargs):
#         acc = 0
#         sample_num = 0
        
#         vqa_score = VQAScore()
        
#         self.to_eval_mode()
        
#         with torch.no_grad():
#             pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
#             for batch_index, (x, y) in pbar:
#                 for k, v in x.items():
#                     if isinstance(v, torch.Tensor):
#                         x[k] = v.to(self.device)
#                 y = y.to(self.device)
#                 output = self.infer(x)
                
#                 vqa_score.update(output, y)
                
#                 pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_acc: {vqa_score.compute():.4f}')

#         return vqa_score.compute()
    
#     def infer(self, x, *args, **kwargs):
#         return self.models_dict['main'](**x)

class ElasticDNN_OfflineVQAMDModel(ElasticDNN_OfflineMDModel):
    # def __init__(self, name: str, models_dict_path: str, device: str, class_to_label_idx_map):
    #     super().__init__(name, models_dict_path, device)
    #     self.class_to_label_idx_map = class_to_label_idx_map
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        vqa_score = VQAScore()
        
        self.to_eval_mode()
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained")
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y, t) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                if isinstance(y,dict):
                    for k, v in y.items():
                        y[k] = v.to(self.device)
                else:
                    y = y.to(self.device)
                
                output = self.models_dict['main'].generate(**x)
                total = 0
                idx = 0
                for i in output:
                    val = processor.decode(i, skip_special_tokens=True)
                    text = t[idx]
                    if val == text:
                        total += 1
                    idx += 1
                    
                #vqa_score.update(output, y.labels)
                acc = total / (idx+1)
                #pbar.set_description(f'cur_batch_total: {len(y['label'])}, cur_batch_acc: {vqa_score.compute():.4f}')
                pbar.set_description(f'cur_batch_total: {len(y["labels"])}, cur_batch_acc: {acc:.4f}')
        #return vqa_score.compute()
        return acc
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'].generate(**x)
    