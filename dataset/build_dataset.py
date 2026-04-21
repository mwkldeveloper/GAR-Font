from dataset.dataset_tokenizer import build_tokenizer_datasets
from dataset.dataset_generator_pre import build_generator_PRE_datasets
from dataset.dataset_generator_nfa import build_generator_NFA_datasets
from dataset.dataset_generator_se import build_generator_SE_datasets
from dataset.dataset_generator_adapter import build_generator_adapter_datasets


def build_dataset(args, **kwargs):
    if args.dataset == 'tokenizer':
        return build_tokenizer_datasets(args, **kwargs)  
    if args.dataset == 'generator_pre':
        return build_generator_PRE_datasets(args, **kwargs)  
    if args.dataset == 'generator_nfa':
        return build_generator_NFA_datasets(args, **kwargs)  
    if args.dataset == 'generator_se':
        return build_generator_SE_datasets(args, **kwargs)
    if args.dataset == 'generator_adapter':
        return build_generator_adapter_datasets(args, **kwargs)
    else:
        raise ValueError(f'dataset {args.dataset} is not supported')