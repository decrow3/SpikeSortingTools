import json
import pickle
import numpy as np
import pytest

from motionqc.field import MotionField, SIGN, canonical_text_hash, load_field


def test_validate_interpolate_and_known_shift(tmp_path):
    field=MotionField(np.arange(0,10,.25),np.array([100.,200.]),np.zeros((40,2)),source="synthetic")
    assert field.validate(recording_length_s=10)==[]
    assert np.all(field.at(np.array([1.,2.]),np.array([120.,180.]))==0)
    manifest=field.save(tmp_path/"field.npz")
    loaded=load_field(tmp_path/"field.npz")
    assert loaded.sign==SIGN
    assert np.array_equal(loaded.displacement_um,field.displacement_um)
    first=(tmp_path/"field.npz").read_bytes();field.save(tmp_path/"again.npz")
    assert first==(tmp_path/"again.npz").read_bytes()
    assert manifest["npz_sha256"]


def test_validate_rejects_bad_axes():
    with pytest.raises(ValueError,match="strictly increasing"):
        MotionField([0,1,1],[0],np.zeros((3,1))).validate()
    with pytest.raises(ValueError,match="uniform"):
        MotionField([0,1,3],[0],np.zeros((3,1))).validate()
    with pytest.raises(ValueError,match="units"):
        MotionField([0,1],[0],np.zeros((2,1)),time_unit="ms").validate()


def test_known_ramp_interpolates_exactly():
    time=np.arange(0,10,.25);ramp=(2*time)[:,None]
    field=MotionField(time,[500],ramp)
    query=np.array([1.125,4.875,8.125])
    assert np.allclose(field.at(query,np.full(3,500)),2*query)


def test_canonical_text_hash_normalizes_line_endings(tmp_path):
    lf=tmp_path/"lf.csv";crlf=tmp_path/"crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n");crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert canonical_text_hash(lf)==canonical_text_hash(crlf)


@pytest.mark.parametrize("keys",[
    ("time_s","depth_um","displacement_um"),
    ("time_bin_centers_s","spatial_bin_centers_um","displacement"),
])
def test_npz_loader_layouts(tmp_path,keys):
    path=tmp_path/(keys[0]+".npz")
    np.savez(path,**{keys[0]:[0.,1.],keys[1]:[100.,200.],keys[2]:np.zeros((2,2))})
    assert load_field(path).displacement_um.shape==(2,2)


def test_raw_directory_and_npx_origin_loader(tmp_path):
    directory=tmp_path/"raw";directory.mkdir()
    np.save(directory/"time_bins.npy",[3057.6775463558583,3058.6775463558583])
    np.save(directory/"depth_bins.npy",[100.])
    np.save(directory/"motion.npy",np.zeros((2,1)))
    field=load_field(directory,format="npx_dredge")
    assert np.allclose(field.time_s,[0.,1.])


def test_pickle_mapping_loader(tmp_path):
    path=tmp_path/"motion.pkl"
    with path.open("wb") as stream:
        pickle.dump({"temporal_bins_s":[0.,1.],"spatial_bins_um":[100.],"motion":np.zeros((2,1))},stream)
    assert load_field(path).displacement_um.shape==(2,1)
