"""The immutable per-run context every stage reads: frozen transcript with
refined word bounds, audio analysis, the judge, and persisted decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import Envelope, RefinedWord
from .config import Config
from .judge import Judge, LLMCache
from .semantic import Sentence
from .timeline import SourceInfo
from .transcribe import Word


class NeedsDecision(RuntimeError):
    """A structural decision had no supermajority; the user must supply it.

    The message names the flag to pass on the rerun; the value is persisted
    in decisions.json so the question is never asked twice.
    """

    def __init__(self, key: str, flag: str, message: str, candidates: list[dict]) -> None:
        super().__init__(message)
        self.key = key
        self.flag = flag
        self.candidates = candidates


@dataclass
class Context:
    config: Config
    source: SourceInfo
    words: list[RefinedWord]
    env: Envelope
    floor: float
    silences: list[tuple[float, float]]
    moans: list[tuple[float, float]]
    judge: Judge
    llm_cache: LLMCache | None
    guidelines: str
    topic: str | None = None
    decisions: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    # Approved owner preferences from this cook's snapshot; empty means none.
    preferences: str = ""
    noise_cuts: list = field(default_factory=list)
    # Voiced, speech-level spans no refined word covers (a word the ASR
    # dropped). The waveform's own word list: the edge guard and the
    # verification report both read it.
    voices: list[tuple[float, float]] = field(default_factory=list)
    # Focused second opinions on those spans, longest first:
    # [{"start", "end", "backend", "text"}]. Evidence for the report, never a
    # source of cuts.
    speech_probes: list[dict] = field(default_factory=list)
    # Whole intervals the cuts must not touch: laughter and unrecognised
    # sound tags, grown over their quiet onset and decay. Known sigh/breath/
    # mouth-noise tags are cut candidates in `noise_cuts`, never listed here.
    protected_spans: list[tuple[float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.source.duration

    def as_words(self) -> list[Word]:
        """Refined words as the transcribe Word type (refined times)."""
        return [Word(w.text, w.start, w.end, 1.0) for w in self.words]

    def sentences(self) -> list[Sentence]:
        from .candidates import sentence_table

        return sentence_table(self.as_words(), split_pause=self.config.sentence_split_pause,
                              burst_last=[w.burst_last for w in self.words])

    def sentence_words(self) -> list[list[int]]:
        """Word indices per sentence, aligned with `sentences()`."""
        from .candidates import sentence_groups

        return sentence_groups(self.as_words(), split_pause=self.config.sentence_split_pause,
                               burst_last=[w.burst_last for w in self.words])
