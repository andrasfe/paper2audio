"""Text-to-speech synthesis without system binaries.

Uses the espeak-ng shared library shipped in the `espeakng-loader`
PyPI wheel (no apt/espeak install needed), driven via ctypes in
retrieval mode so raw PCM is collected in-process, then encoded to MP3
with `lameenc`. Works fully offline.
"""
from __future__ import annotations

import ctypes
import pathlib
import re
import wave

# espeak-ng constants
_AUDIO_OUTPUT_RETRIEVAL = 1
_ESPEAK_CHARS_UTF8 = 1
_POS_CHARACTER = 1
_PARAM_RATE = 1
_PARAM_PITCH = 3
_PARAM_WORDGAP = 7

_SynthCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
)


class Synthesizer:
    """espeak-ng wrapper. The shared library supports only ONE
    espeak_Initialize per process (a second call corrupts its heap), so
    use get_synthesizer() rather than constructing this directly."""

    _instance: "Synthesizer | None" = None

    def __init__(self, voice: str = "en-us", rate_wpm: int = 165, pitch: int = 50):
        if Synthesizer._instance is not None:
            raise RuntimeError(
                "espeak-ng can only be initialized once per process; "
                "use tts.get_synthesizer()"
            )
        import espeakng_loader

        self.lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
        data_path = espeakng_loader.get_data_path()
        self.sample_rate = self.lib.espeak_Initialize(
            _AUDIO_OUTPUT_RETRIEVAL, 0, str(data_path).encode(), 0
        )
        if self.sample_rate <= 0:
            raise RuntimeError("espeak_Initialize failed")
        self.configure(voice, rate_wpm, pitch)

        self._pcm = bytearray()

        @_SynthCallback
        def _cb(wav, numsamples, events):
            if numsamples > 0:
                self._pcm += ctypes.string_at(
                    wav, numsamples * ctypes.sizeof(ctypes.c_short)
                )
            return 0

        self._cb = _cb  # keep a reference; ctypes callbacks must outlive use
        self.lib.espeak_SetSynthCallback(self._cb)
        Synthesizer._instance = self

    def configure(self, voice: str, rate_wpm: int, pitch: int = 50) -> None:
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            self.lib.espeak_SetVoiceByName(b"en")
        self.lib.espeak_SetParameter(_PARAM_RATE, rate_wpm, 0)
        self.lib.espeak_SetParameter(_PARAM_PITCH, pitch, 0)
        self.lib.espeak_SetParameter(_PARAM_WORDGAP, 1, 0)

    def synth_chunks(self, chunks: list[str], progress=None) -> bytes:
        """Synthesize text chunks to 16-bit mono PCM at self.sample_rate."""
        self._pcm = bytearray()
        for i, chunk in enumerate(chunks):
            data = chunk.encode("utf-8")
            rc = self.lib.espeak_Synth(
                data, len(data) + 1, 0, _POS_CHARACTER, 0, _ESPEAK_CHARS_UTF8,
                None, None,
            )
            if rc != 0:
                raise RuntimeError(f"espeak_Synth failed on chunk {i} (rc={rc})")
            self.lib.espeak_Synchronize()
            # Short pause between chunks (0.35 s of silence).
            self._pcm += b"\x00" * int(self.sample_rate * 0.35) * 2
            if progress:
                progress(i + 1, len(chunks), len(self._pcm))
        return bytes(self._pcm)


def get_synthesizer(
    voice: str = "en-us", rate_wpm: int = 165, pitch: int = 50
) -> Synthesizer:
    """Return the process-wide synthesizer, (re)configured as requested."""
    if Synthesizer._instance is None:
        return Synthesizer(voice=voice, rate_wpm=rate_wpm, pitch=pitch)
    inst = Synthesizer._instance
    inst.configure(voice, rate_wpm, pitch)
    return inst


def split_paragraphs(text: str, max_chars: int = 4000) -> list[str]:
    """Split narration into paragraph chunks, further splitting any
    paragraph that exceeds max_chars at sentence boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) + 1 > max_chars:
                chunks.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            chunks.append(buf)
    return chunks


def pcm_to_wav(pcm: bytes, sample_rate: int, path: str | pathlib.Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def pcm_to_mp3(
    pcm: bytes, sample_rate: int, path: str | pathlib.Path, bitrate_kbps: int = 64
) -> None:
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate_kbps)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(pcm)
    mp3 += enc.flush()
    pathlib.Path(path).write_bytes(bytes(mp3))


def text_to_mp3(
    text: str,
    out_path: str | pathlib.Path,
    voice: str = "en-us",
    rate_wpm: int = 165,
    bitrate_kbps: int = 64,
    progress=None,
) -> float:
    """Full pipeline: narration text -> MP3 file. Returns duration in seconds."""
    synth = get_synthesizer(voice=voice, rate_wpm=rate_wpm)
    chunks = split_paragraphs(text)
    pcm = synth.synth_chunks(chunks, progress=progress)
    pcm_to_mp3(pcm, synth.sample_rate, out_path, bitrate_kbps)
    return len(pcm) / 2 / synth.sample_rate
