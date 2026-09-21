from typing import Tuple

import numpy as np

from .base import DatasetItem, Task


class MultiplicationTask(Task):
    """
    Generator for decimal multiplication.

    Designed for length-generalization experiments where the length of the
    first operand changes while the second operand remains short.

    Typical setup:

        training / ID:
            first operand  <= 5 digits
            second operand <= 3 digits

        OOD:
            first operand  = 35 digits
            second operand <= 3 digits

    Prompt layout:

        a_0 ... a_{d-1}  MUL  b_0 ... b_{m-1}  EQ

    Answer layout:

        p_0 ... p_{k-1}

    Numbers are emitted most-significant-digit first.

    Unlike AdditionTask, operands are not padded internally. Padding of the
    packed sequence is handled by Task.collate().
    """

    PAD_ID = 0
    MUL_ID = 1
    EQ_ID = 2
    N_SPECIAL = 3

    BASE = 10

    def __init__(
        self,
        n_digits: int | Tuple[int, int],
        multiplier_digits: int = 3,
        exact_digits: bool = True,
        fixed_answer_width: bool = True,
        seed: int | None = 42,
    ):
        """
        Parameters
        ----------
        n_digits:
            Length of the first operand.

            If int:
                exact_digits=True:
                    sample exactly n_digits digits.

                exact_digits=False:
                    sample a positive integer with at most n_digits digits.

            If tuple (lo, hi):
                sample the first-operand length uniformly from [lo, hi).

        multiplier_digits:
            Maximum number of digits of the second operand.

            The second operand is sampled uniformly from

                1 <= b < 10**multiplier_digits.

            Thus multiplier_digits=3 corresponds to b in [1, 999].

        exact_digits:
            Controls the integer n_digits case only.

            True:
                n_digits=5 -> exactly 5-digit first operands.

            False:
                n_digits=5 -> first operands with at most 5 digits.

            For the paper-style short training distribution, use False.

            For a controlled 35-digit OOD set, use True.

        fixed_answer_width:
            If True, left-pad the product with zero digits to the maximum
            possible product width:

                max first-operand digits + multiplier_digits.

            This prevents answer length from leaking information about the
            numerical result and makes answer length deterministic for a
            given task configuration.

        seed:
            RNG seed.
        """
        self.n_digits = n_digits
        self.multiplier_digits = multiplier_digits
        self.exact_digits = exact_digits
        self.fixed_answer_width = fixed_answer_width

        if multiplier_digits < 1:
            raise ValueError("multiplier_digits must be >= 1")

        if isinstance(n_digits, int):
            if n_digits < 1:
                raise ValueError("n_digits must be >= 1")
        else:
            lo, hi = n_digits
            if lo < 1 or hi <= lo:
                raise ValueError(
                    "n_digits tuple must satisfy 1 <= lo < hi"
                )

        self.n_special = self.N_SPECIAL
        self.d_token_ids = (
            np.arange(self.BASE, dtype=np.int64) + self.n_special
        )

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.n_special + self.BASE

    @property
    def max_digits(self) -> int:
        """Maximum possible length of the first operand."""
        if isinstance(self.n_digits, int):
            return self.n_digits

        return self.n_digits[1] - 1

    @property
    def max_answer_digits(self) -> int:
        """
        Maximum product width.

        A d-digit number times an m-digit number has at most d + m digits.
        """
        return self.max_digits + self.multiplier_digits

    @property
    def min_block_size(self) -> int:
        """
        Smallest block_size accepted by Task.collate().

        Longest prompt:

            d + 1 + m + 1
              = d + m + 2

        where the two extra tokens are MUL and EQ.

        Longest answer:

            d + m

        Therefore

            len(prompt) + len(answer)
                = 2d + 2m + 2

        and base.py requires

            len(prompt) + len(answer) <= block_size + 1.
        """
        max_sequence_length = (
            2 * self.max_digits
            + 2 * self.multiplier_digits
            + 2
        )

        return max_sequence_length - 1

    def metrics(self, predicted, targets):
        """
        Exact-match accuracy plus token-level answer accuracy.
        """
        scores = super().metrics(predicted, targets)

        answer_mask = targets != -1

        if answer_mask.any():
            scores["digit_acc"] = float(
                (predicted == targets)[answer_mask]
                .astype(np.float32)
                .mean()
            )
        else:
            scores["digit_acc"] = 0.0

        return scores

    def _sample_first_length(self) -> int:
        """Choose the number of digits of the first operand."""
        if isinstance(self.n_digits, tuple):
            lo, hi = self.n_digits
            return int(self.rng.integers(lo, hi))

        if self.exact_digits:
            return self.n_digits

        # Sampling the integer itself uniformly below 10**n is the
        # paper-style distribution. Its digit length is therefore not
        # uniform over 1,...,n.
        return -1

    def _sample_first_operand(self) -> int:
        """Sample the first operand."""
        d = self._sample_first_length()

        if d == -1:
            # At most self.n_digits digits.
            return int(
                self.rng.integers(
                    1,
                    10**self.n_digits,
                )
            )

        lower = 1 if d == 1 else 10 ** (d - 1)
        upper = 10**d

        return int(
            self.rng.integers(
                lower,
                upper,
            )
        )

    def _sample_second_operand(self) -> int:
        """
        Sample the short second operand.

        multiplier_digits=3 gives a uniform draw from 1,...,999.
        """
        return int(
            self.rng.integers(
                1,
                10**self.multiplier_digits,
            )
        )

    @staticmethod
    def _digits(value: int) -> np.ndarray:
        """Positive integer -> MSB-first decimal digit array."""
        if value <= 0:
            raise ValueError("Expected a positive integer")

        return np.fromiter(
            (ord(c) - ord("0") for c in str(value)),
            dtype=np.int64,
        )

    def _encode_digits(self, digits: np.ndarray) -> np.ndarray:
        """Decimal digits -> vocabulary token IDs."""
        return self.d_token_ids[digits].astype(np.int64)

    def _encode_number(self, value: int) -> np.ndarray:
        """Positive integer -> MSB-first token sequence."""
        return self._encode_digits(self._digits(value))

    def _encode_product(self, product: int) -> np.ndarray:
        """
        Encode the multiplication result.

        When fixed_answer_width=True, prepend zero DIGITS until the answer
        has max_answer_digits positions.

        These are digit-zero tokens, not PAD tokens, because they are part
        of the supervised answer.
        """
        digits = self._digits(product)

        if self.fixed_answer_width:
            width = self.max_answer_digits

            if len(digits) > width:
                raise ValueError(
                    f"Product needs {len(digits)} digits but configured "
                    f"answer width is only {width}"
                )

            if len(digits) < width:
                digits = np.concatenate(
                    [
                        np.zeros(
                            width - len(digits),
                            dtype=np.int64,
                        ),
                        digits,
                    ]
                )

        return self._encode_digits(digits)

    def _sample_one(self) -> DatasetItem:
        a = self._sample_first_operand()
        b = self._sample_second_operand()

        product = a * b

        a_tokens = self._encode_number(a)
        b_tokens = self._encode_number(b)

        prompt = np.concatenate(
            [
                a_tokens,
                np.array([self.MUL_ID], dtype=np.int64),
                b_tokens,
                np.array([self.EQ_ID], dtype=np.int64),
            ]
        )

        answer = self._encode_product(product)

        return DatasetItem(
            prompt=prompt.astype(np.int64),
            answer=answer.astype(np.int64),
            metadata={
                "a": a,
                "b": b,
                "product": product,
                "n_digits": len(str(a)),
                "multiplier_digits": len(str(b)),
                "max_digits": self.max_digits,
                "max_multiplier_digits": self.multiplier_digits,
                "length_group": (
                    "LONG"
                    if len(str(a)) > 5
                    else "SHORT"
                ),
                "variant": "MULT",
            },
        )
