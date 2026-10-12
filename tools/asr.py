import sys, json
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
import numpy as np; a = np.fromfile(sys.argv[1], dtype=np.float32)
segs, _ = m.transcribe(a, language="ru", word_timestamps=True)
words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for s in segs for w in s.words]
json.dump(words, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(" ".join(f"{x['w']}@{x['s']}" for x in words))
