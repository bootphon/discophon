import importlib
import itertools
import math
import pickle
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal, Optional

import panphon
import torch
import torchaudio
from jaxtyping import Float, Integer, Num
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .decoder import FeatureDecoder, SILENCE
from .io import read_alignment, read_manifest
from .utils import MyPathLike

SAMPLE_RATE = 16_000
ALIGNMENT_FREQ = 100  # in Hz
MODEL_FREQ = 50  # in Hz
FEATURE_SIZE = 0.02  # 1 / MODEL_FREQ, in seconds
SUBSAMPLE = ALIGNMENT_FREQ // MODEL_FREQ
CONV_KERNEL_STRIDE = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]


class FeatureTokenizer:
    """FeatureTokenizer is a class that handles tokenization and encoding of IPA
    (International Phonetic Alphabet) phones into feature representations using a
    FeatureDecoder.

    Attributes
    ----------
    PAD_INDEX : int
        Index used for padding.

    Methods
    -------
    __init__(unknown_mode, feature_decoder):
        Initializes the FeatureTokenizer with a specific unknown mode and a feature
        decoder.

    unknown_mode:
        Property to access the unknown mode setting.

    num_features:
        Property to get the total number of features as per feature decoder header.

    unknown_index:
        Property to compute the index for unknown tokens based on the unknown mode.

    multilingual_mode:
        Property to check if multilingual mode is enabled in the feature decoder.

    ipa_to_features(ipa_phone):
        Converts an IPA phone to its representative features.

    encode(ipa_phones, counts):
        Encodes a sequence of IPA phones along with their counts into tensors.

    decode(tokens):
        Abstract method meant to decode feature tensors back into sequence of IPA
        phones.

    Notes
    -----
    Positive features are encoded as 1, negative features are encoded as 0, and zero
    features are encoded as 0 (if unknown mode is "no-unk") or 2 (otherwise).
    """
    PAD_INDEX: int = 3

    def __init__(
            self,
            unknown_mode: Literal["no-unk", "unk-pred", "unk-ign"],
            feature_decoder: FeatureDecoder,
    ) -> None:
        self._unknown_mode = unknown_mode
        self._feat_decoder = feature_decoder
        self._ipa_to_feats = {seg: feats for seg, feats in
                              zip(feature_decoder.segments, feature_decoder.features)}

    @property
    def unknown_mode(self) -> Literal["no-unk", "unk-pred", "unk-ign"]:
        return self._unknown_mode

    @property
    def num_features(self) -> int:
        return len(self._feat_decoder.header)

    @property
    def unknown_index(self) -> int:
        return 0 if self.unknown_mode == "no-unk" else 2

    @property
    def multilingual_mode(self) -> bool:
        return self._feat_decoder.multilingual_mode

    @cache
    def ipa_to_features(self, ipa_phone: str) \
            -> tuple[tuple[str, ...], Num[torch.Tensor, "_ {self.num_features}"]]:
        """Get the representative form and the feature representation of an IPA phone.

        Parameters
        ----------
        ipa_phone: str
            The IPA phone to convert.

        Returns
        -------
        phone_strings : tuple of str
            A tuple with the representative phones.
        feature_tensor : torch.Tensor
            Tensor representing the features of the IPA phone.
        """
        rep_phones = self._feat_decoder.segment_to_representative(ipa_phone)
        rep_phones, vector = self._feat_decoder.canonical_representation(rep_phones)

        dtype = torch.long if self.unknown_mode == "unk-pred" else torch.float
        tensor = torch.from_numpy(vector).type(dtype)
        tensor[tensor == 0] = self.unknown_index
        tensor[tensor < 0] = 0

        return rep_phones, tensor

    def encode(
            self,
            ipa_phones: tuple[str, ...],
            counts: tuple[int, ...]
    ) -> tuple[torch.Tensor, list[str]]:
        """Encode a sequence of IPA phones along with their counts into tensors.

        Parameters
        ----------
        ipa_phones : tuple of str
            A tuple of IPA symbols representing phones.
        counts : tuple of int
            A tuple of integers representing the repetition count of each phone.

        Returns
        -------
        feature_tensor : torch.Tensor
            The tensor containing the encoded features.
        phones : list of str
            A list of phone strings.
        """
        assert len(counts) == len(ipa_phones), \
            (f"Length mismatch between the IPA phones ({len(ipa_phones)}) and counts "
             f"({len(counts)})")
        vectors = []
        phones = []
        for phone, reps in zip(ipa_phones, counts):
            phs, vec = self.ipa_to_features(phone)
            if len(vec) == 1:
                vectors.append(vec.repeat(reps, 1))
                phones += [phs[0]] * reps
            else:
                boundaries = [round(i * reps / len(vec)) for i in range(len(vec) + 1)]
                lengths = [e - b for b, e in zip(boundaries[:-1], boundaries[1:])]
                vectors.append(vec.repeat_interleave(torch.tensor(lengths), dim=0))
                phones += [ph for ph, len_ in zip(phs, lengths) for _ in range(len_)]
        return torch.cat(vectors, dim=0), phones

    def decode(self, tokens: Num[torch.Tensor, "seq {self.num_features}"]) -> list[str]:
        """Decode feature tensors back into a sequence of IPA phones.

        Parameters
        ----------
        tokens : torch.Tensor
            The tensor containing the encoded features.

        Returns
        -------
        ipa_phones : list of str
            A list of IPA phone strings.
        """
        raise NotImplementedError


class Tokenizer:
    """Tokenizer is a class that handles tokenization and encoding of IPA
    (International Phonetic Alphabet) phones into indices from the list of segments
     in a FeatureDecoder.

    Methods
    -------
    __init__(unknown_mode, feature_decoder):
        Initializes the FeatureTokenizer with a specific unknown mode and a feature
        decoder.

    sum_diphthong:
        Property to know whether the diphthongs have been merged..

    multilingual_mode:
        Property to check if multilingual mode is enabled in the feature decoder.

    pad_token:
        Property to access the token used for padding.

    ipa_to_features(ipa_phone):
        Converts an IPA phone into token indices.

    encode(ipa_phones, counts):
        Encodes a sequence of IPA phones along with their counts into tensors.

    decode(tokens):
        Decode token tensors back into sequence of IPA phones.
    """

    def __init__(self, feature_decoder: FeatureDecoder, inventory: list[str]) -> None:
        self._feat_decoder = feature_decoder
        self._inventory = inventory
        self._ipa_to_feats = {seg: feats for seg, feats in
                              zip(feature_decoder.segments, feature_decoder.features)}

    @property
    def sum_diphthong(self):
        return self._feat_decoder.unique_seg_feats.sum_diphthong

    @property
    def multilingual_mode(self) -> bool:
        return self._feat_decoder.multilingual_mode

    @property
    def pad_token(self) -> int:
        return len(self._feat_decoder.segments)

    @cache
    def ipa_to_token(self, ipa_phone: str) \
            -> tuple[tuple[str, ...], list[int]]:
        """Get the representative form and the token (index) of an IPA phone.

        Parameters
        ----------
        ipa_phone: str
            The IPA phone to convert.

        Returns
        -------
        phone_strings : tuple of str
            A tuple with the representative phones.
        feature_tensor : torch.Tensor
            Tensor representing the IPA phone tokens.
        """
        rep_phones = self._feat_decoder.segment_to_representative(ipa_phone)
        tokens = [self._inventory.index(phone) for phone in rep_phones]
        return rep_phones, tokens

    def encode(
            self, ipa_phones: tuple[str, ...], counts: tuple[int, ...]
    ) -> torch.LongTensor:
        """Encode a sequence of IPA phones into tensors.
        Parameters
        ----------
        ipa_phones : tuple of str
            A tuple of IPA symbols representing phones.
        counts : tuple of int
            A tuple of integers representing the repetition count of each phone.

        Returns
        -------
        tokens : torch.Tensor
            The tensor containing the phone tokens.
        """
        assert len(counts) == len(ipa_phones), \
            (f"Length mismatch between the IPA phones ({len(ipa_phones)}) and counts "
             f"({len(counts)})")

        tokens = []
        for phone, reps in zip(ipa_phones, counts):
            toks = self.ipa_to_token(phone)[1]
            if len(toks) == 1:
                tokens += [toks[0]] * reps
            else:
                boundaries = [round(i * reps / len(toks)) for i in range(len(toks) + 1)]
                lengths = [e - b for b, e in zip(boundaries[:-1], boundaries[1:])]
                tokens += [tok for tok, len_ in zip(toks, lengths) for _ in range(len_)]
        return torch.LongTensor(tokens)

    def decode(self, tokens: Integer[torch.Tensor, "seq"]) -> str:
        """Decode token tensors back into a sequence of IPA phones.

        Parameters
        ----------
        tokens : torch.Tensor
            The tensor containing the tokens.

        Returns
        -------
        ipa_phones : str
            A concatenation of IPA phone strings separated by whitespace.
        """
        return " ".join(
            self._inventory[token] for token in tokens if token != self.pad_token
        )


class Librispeech:
    LIBRI_TO_IPA: dict[str, str] = {  # no AX, AXR, DX, IX, NX, Q, WH
        "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b",
        "CH": "t͡ʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜ˞", "EY": "eɪ", "F": "f",
        "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "d͡ʒ", "K": "k", "L": "l",
        "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ",
        "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
        "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ", "SIL": SILENCE,
    }

    @classmethod
    def convert_to_ipa(cls, libri_phones: list[str] | str) -> list[str]:
        if isinstance(libri_phones, str):
            libri_phones = libri_phones.split(" ")
        return [cls.LIBRI_TO_IPA[phone] for phone in libri_phones]


class MontrealForcedAligner:
    MFA_TO_IPA: dict[str, str] = {"g": "ɡ", "ɚ": "ə˞", "ʡ": "ʔ"}

    @classmethod
    def convert_to_ipa(cls, montreal_phones: list[str] | str) -> list[str]:
        if isinstance(montreal_phones, str):
            montreal_phones = montreal_phones.split(" ")
        return [cls.MFA_TO_IPA.get(phone, phone) for phone in montreal_phones]


class PanPhonInventory:
    def __init__(self):
        with open(
                importlib.resources.files(
                    "phind"
                ) / "../../data" / "correction_map.pickle",
                "rb"
        ) as fp:
            self._corrections = pickle.load(fp)

    def convert_to_ipa(self, panphon_phones: list[str] | str) -> list[str]:
        if isinstance(panphon_phones, str):
            panphon_phones = panphon_phones.split(" ")
        return [self._corrections.get(phone, phone) for phone in panphon_phones]


def conv_output_length(input_length: int):
    """Compute the output length of a sequence after the series of 1D convolutions in
    HuBERT's feature extractor."""
    for kernel_size, stride in CONV_KERNEL_STRIDE:
        input_length = math.floor((input_length - kernel_size) / stride + 1)
    return input_length


def subsample_tokens(
        tokens: list, num_samples: int, pad_token: Optional[Any] = None
) -> list:
    """Subsample the phone sequence to match the output length of HuBERT."""
    subsampled = tokens[::SUBSAMPLE]
    output_length = conv_output_length(num_samples)
    if len(subsampled) >= output_length:
        return subsampled[:output_length]
    assert pad_token is not None
    return subsampled + [pad_token] * (output_length - len(subsampled))


class IPADataset:
    def __init__(
            self,
            manifest_path: MyPathLike,
            alignments_path: MyPathLike,
            sep_diphthong: bool = False
    ) -> None:
        manifest = read_manifest(manifest_path)
        self.manifest = list(manifest.items())
        alignments = read_alignment(alignments_path)
        self.ipa_phones = {
            file_id: Librispeech.convert_to_ipa(phones)
            for file_id, phones in alignments.items()
        }
        self.sep_diphthong = sep_diphthong
        self._ft = panphon.FeatureTable()

    def is_diphthong(self, phone: str) -> bool:
        return len(self._ft.word_to_vector_list(phone)) > 1

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int]]:
        if idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range")
        file_id, (_, num_samples) = self.manifest[idx]
        phones = subsample_tokens(self.ipa_phones[file_id], num_samples)
        ipa_phones, counts = [], []
        for phone, group in itertools.groupby(phones):
            reps = len(list(group))
            if self.sep_diphthong and self.is_diphthong(phone):
                l1 = math.ceil(reps / 2)
                ipa_phones += phone
                counts += [l1, reps - l1]
            else:
                ipa_phones.append(phone)
                counts.append(reps)
        return ipa_phones, counts


class PhoneticFeatureDataset(Dataset):
    def __init__(
            self,
            manifest_path: MyPathLike,
            alignment_path: MyPathLike,
            feature_tokenizer: FeatureTokenizer,
            separate_files: bool = False,
            max_audio_length: int = 28_800_000,  # for backwards compatibility
            pad_audio: bool = True,  # for backwards compatibility
            random_crop: bool = False,
            boundary_prediction: bool = False,
            silence_prediction: bool = False
    ) -> None:
        super().__init__()
        if boundary_prediction or silence_prediction:
            assert feature_tokenizer.unknown_mode in ["no-unk", "unk-ign"], \
                (
                    "Boundary and silence predictions are only compatible with unknown "
                    "modes 'no-unk' and 'unk-ign', but got "
                    f"'{feature_tokenizer.unknown_mode}'"
                )

        self.max_audio_length = max_audio_length
        self.pad_audio = pad_audio
        self.random_crop = False  # random_crop
        self.boundary_prediction = boundary_prediction
        self.silence_prediction = silence_prediction
        self.feature_tokenizer = feature_tokenizer
        panphon_inventory = PanPhonInventory()
        if separate_files:
            manifests = [read_manifest(man_path) for man_path in
                         Path(manifest_path).glob("¡*.tsv")]
            self.manifest = [entry for man in manifests for entry in man.items()]
            alignments = [read_alignment(align_path) for align_path in
                          Path(alignment_path).glob("*.align")]
            self.ipa_phones: dict[str, list[str]] = {}
            for align in alignments:
                self.ipa_phones.update(
                    {file: panphon_inventory.convert_to_ipa(_align) for file, _align in
                     align.items()}
                )

            self.lang_sizes = [len(man) for man in manifests]
        else:
            manifest = read_manifest(manifest_path)
            self.manifest = list(manifest.items())
            alignments = read_alignment(alignment_path)
            if feature_tokenizer.multilingual_mode:
                self.ipa_phones = {file: panphon_inventory.convert_to_ipa(_align) for
                                   file, _align in alignments.items()}
            else:
                self.ipa_phones = {
                    file_id: Librispeech.convert_to_ipa(phones)
                    for file_id, phones in alignments.items()
                }

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]:
        if idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range")

        file_id, (path, num_samples) = self.manifest[idx]
        waveform, sample_rate = torchaudio.load(path)
        assert waveform.size(0) == 1
        waveform = waveform.squeeze(0)
        assert sample_rate == SAMPLE_RATE and waveform.size(0) == num_samples
        phones = subsample_tokens(self.ipa_phones[file_id], num_samples, SILENCE)
        phones, counts = zip(
            *[(ph, len(list(gr))) for ph, gr in itertools.groupby(phones)]
        )
        phon_features, phones = self.feature_tokenizer.encode(phones, counts)
        counts = torch.as_tensor(counts)

        additions = []
        if self.boundary_prediction:
            count_mask = counts.cumsum(dim=0)[:-1]
            count_features = phon_features.new_zeros(phon_features.size(0), 1)
            count_features[count_mask] = 1
            additions.append(count_features)
        if self.silence_prediction:
            sil_features = phon_features.new_zeros(phon_features.size(0), 1)
            sil_mask = [ph == self.feature_tokenizer.ipa_to_features(SILENCE)[0][0]
                        for ph in phones]
            sil_features[sil_mask] = 1
            additions.append(sil_features)

        if additions:
            phon_features = torch.cat([phon_features, *additions], dim=1)
        return waveform, phon_features, counts, phones

    def collate_audios(self, audios: list[torch.Tensor], target_length: int) \
            -> tuple[torch.Tensor, list[int], list[int]]:
        batch = audios[0].new_zeros(len(audios), target_length)
        beginnings = [0] * len(audios)
        lengths = [target_length] * len(audios)
        for i, audio in enumerate(audios):
            diff = audio.size(0) - target_length
            if diff == 0:
                batch[i] = audio
            elif diff < 0:
                assert self.pad_audio
                batch[i, :audio.size(0)] = audio
                lengths[i] = audio.size(0)
            else:
                beg = torch.randint(diff + 1, size=(1,))[0] if self.random_crop else 0
                batch[i] = audio[beg:beg + target_length]
                beginnings[i] = beg
        return batch, beginnings, lengths

    def collate_phonetic_features(
            self,
            phon_features: list[torch.Tensor],
            beginnings: list[int],
            lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = phon_features[0].new_full(
            (len(phon_features), max(lengths), phon_features[0].size(-1)),
            fill_value=self.feature_tokenizer.PAD_INDEX
        )
        for i, feats in enumerate(phon_features):
            batch[i, :lengths[i]] = feats[beginnings[i]:beginnings[i] + lengths[i]]
        return batch, torch.as_tensor(lengths)

    @staticmethod
    def collate_counts(
            counts: list[torch.Tensor], phon_begs: list[int], phon_lengths: list[int]
    ) -> torch.Tensor:
        new_counts = []
        for _counts, beg, length in zip(counts, phon_begs, phon_lengths):
            cumul = _counts.cumsum(dim=0)
            fst_idx = torch.searchsorted(cumul, beg)
            _counts = _counts[fst_idx:].detach().clone()
            _counts[0] = cumul[fst_idx] - beg
            cumul = _counts.cumsum(dim=0)
            idx = torch.searchsorted(cumul, length)
            _counts[idx] = length - (cumul[idx - 1] if idx > 0 else 0)
            new_counts.append(_counts[:idx + 1])
        return pad_sequence(new_counts, batch_first=True)

    def collate_fn(
            self,
            batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]]
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, tuple[Sequence[str], ...]
    ]:
        waveforms, phon_features, counts, phones = zip(*batch)

        waveform_lengths = [len(w) for w in waveforms]
        if self.pad_audio:
            audio_length = min(max(waveform_lengths), self.max_audio_length)
        else:
            audio_length = min(min(waveform_lengths), self.max_audio_length)
        waveforms, waveform_beginnings, waveform_lengths = self.collate_audios(
            waveforms, audio_length
        )
        # TODO: how to handle the mismatch between the waveform shift and the
        #  waveform-phone alignment (20ms frames) for random cropping?
        # actual phonetic sequence lengths
        phon_feature_beginnings = [max(0, conv_output_length(wb + 1)) for wb in
                                   waveform_beginnings]
        phon_feature_lengths = [conv_output_length(wl) for wl in waveform_lengths]
        phones = tuple(
            phn_seq[beg:beg + length] for phn_seq, beg, length in
            zip(phones, phon_feature_beginnings, phon_feature_lengths)
        )
        counts = self.collate_counts(
            counts, phon_feature_beginnings, phon_feature_lengths
        )
        waveform_lengths = torch.as_tensor(waveform_lengths)
        phon_features, phon_feature_lengths = self.collate_phonetic_features(
            phon_features, phon_feature_beginnings, phon_feature_lengths
        )
        return (waveforms, phon_features, counts, waveform_lengths,
                phon_feature_lengths, phones)


class AudioLabelDataset(Dataset):
    def __init__(
            self,
            manifest_path: MyPathLike,
            labels_path: MyPathLike
    ) -> None:
        super().__init__()
        manifest = read_manifest(manifest_path)
        self.manifest = list(manifest.items())
        with open(labels_path, "r") as fp:
            self.labels = [line.rstrip() for line in fp]
        assert len(self.manifest) == len(self.labels), \
            f"{len(self.manifest)} in manifest vs {len(self.labels)} labelled "

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) \
            -> tuple[Float[torch.Tensor, "length"], Integer[torch.Tensor, "seq"]]:
        if idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range")

        _, (path, num_samples) = self.manifest[idx]
        waveform, sample_rate = torchaudio.load(path)
        assert waveform.ndim == 2 and waveform.size(0) == 1
        assert waveform.size(1) == num_samples and sample_rate == SAMPLE_RATE
        waveform = waveform.squeeze(0)
        label = torch.as_tensor([int(el) for el in self.labels[idx].split()])
        return waveform, label

    @staticmethod
    def collate_fn(
            batch: list[
                tuple[Float[torch.Tensor, "length"], Integer[torch.Tensor, "seq"]]
            ]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveforms, labels = zip(*batch)
        waveform_lengths = torch.as_tensor([len(w) for w in waveforms])
        waveforms = pad_sequence(waveforms, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)
        return waveforms, labels, waveform_lengths
