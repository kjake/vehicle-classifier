cd $(dirname $0)
if [ ! -f .venv/bin/activate ]
then
    python3 -m venv .venv
fi

. .venv/bin/activate
# needed for dataset export and model export
pip install datasets openvino coremltools ncnn pnnx onnx
