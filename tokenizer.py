

import collections
import json
import os
import re
from typing import List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json'}
PRETRAINED_VOCAB_FILES_MAP = {
    'qm9': {
        'vocab_file': {}
    },
    'zinc250k': {
        'vocab_file': {}
    }
}


class SMILESTokenizer(PreTrainedTokenizer):


    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token='<unk>',
        sep_token='<eos>',
        pad_token='<pad>',
        cls_token='<bos>',
        mask_token='<mask>',
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path"
                f"'{vocab_file}'."
            )
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            vocab_from_file = json.load(vocab_handle)

        self.vocab = {
            cls_token: 0,
            sep_token: 1,
            mask_token: 2,
            pad_token: 3,
            unk_token: 4,
            **{k: v + 5 for k, v in vocab_from_file.items()}
        }

        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])


        self.pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex_tokenizer = re.compile(self.pattern)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def _convert_token_to_id(self, token):

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):

        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):

        out_string = "".join(tokens).strip()
        return out_string


    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:


        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep


    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:


        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:


        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(
        self, save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Union[Tuple[str],  None]:
        if not os.path.isdir(save_directory):
            logger.error(
                f"Vocabulary path ({save_directory}) should"
                "be a directory.")
            return None
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    self.vocab,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False
                ) + "\n")
        return (vocab_file,)


class QM9Tokenizer(SMILESTokenizer):
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP['qm9']


class Zinc250kTokenizer(SMILESTokenizer):
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP['zinc250k']
