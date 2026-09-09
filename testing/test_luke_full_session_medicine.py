import hashlib
import json

import numpy as np
import pytest

from testing.luke_full_session_medicine import (
    concatenate_chunks, padded_read, sample_chunks, save, seal, sealed,
)


def test_full_session_sample_partition_has_no_gaps_overlaps_or_lost_tail():
    fs, total = 29999.835983263598, 314204894
    chunks = sample_chunks(total, fs)
    assert chunks[0][0] == 0 and chunks[-1][1] == total
    assert all(a < b for a, b in chunks)
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))
    assert sum(b-a for a, b in chunks) == total
    assert chunks[-1][1]-chunks[-1][0] < round(20*fs)
    assert any(a == round(930*fs) for a, _ in chunks)


def test_read_padding_never_wraps_and_streamed_hash_is_source_hash(tmp_path):
    data = np.arange(63*3, dtype='<i2').reshape(63, 3)
    raw = tmp_path/'raw.bin'; raw.write_bytes(data.tobytes())
    h = hashlib.sha256()
    for start, stop in [(0, 10), (10, 30), (30, 50), (50, 63)]:
        x, core, edge = padded_read(raw, start, stop, 63, 3, 4)
        assert len(x) == stop-start+8
        np.testing.assert_array_equal(x[4:-4], data[start:stop])
        h.update(core)
        if start == 0:
            np.testing.assert_array_equal(x[:4], data[4:0:-1])
            assert edge == [4, 0]
        if stop == 63:
            np.testing.assert_array_equal(x[-4:], data[-2:-6:-1])
            assert edge == [0, 4]
    assert h.hexdigest() == hashlib.sha256(data.tobytes()).hexdigest()


def test_merge_keeps_absolute_samples_and_empty_intervals(tmp_path):
    dtype = [('sample_index', 'i8'), ('channel_index', 'i8'), ('amplitude', 'f8'), ('segment_index', 'i8')]
    stages = []
    for i, samples in enumerate([[1, 9], [], [2, 8]]):
        path = tmp_path/str(i); path.mkdir(); stages.append(path)
        p = np.zeros(len(samples), dtype=dtype); p['sample_index'] = samples
        y = np.zeros(len(samples), dtype=[('y', 'f8')]); y['y'] = 100+i
        np.save(path/'peaks.npy', p); np.save(path/'locations.npy', y)
        save(path/'audit.json', dict(start_sample=i*10, stop_sample=(i+1)*10))
    out = tmp_path/'merged'; out.mkdir()
    concatenate_chunks(stages, out, 30, 3.)
    np.testing.assert_array_equal(np.load(out/'peaks.npy')['sample_index'], [1, 9, 22, 28])
    np.testing.assert_array_equal(np.load(out/'locations.npy')['y'], [100, 100, 102, 102])
    assert json.loads((out/'input.json').read_text())['stop_s'] == 10.


def test_unsealed_and_corrupt_evidence_cannot_be_reused(tmp_path):
    stage = tmp_path/'stage'
    assert not sealed(stage)
    stage.mkdir()
    with pytest.raises(RuntimeError, match='Unsealed'):
        sealed(stage)
    (stage/'array').write_bytes(b'original')
    seal(stage); assert sealed(stage)
    (stage/'array').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Invalid sealed'):
        sealed(stage)
