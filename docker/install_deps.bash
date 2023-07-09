#!/user/bin/env bash
echo ""
echo "Start installing python dependencies..."
echo ""

python -m pip install --upgrade pip
python -m pip install -r ${LFD_PKG_PATH}/docker/requirements.txt || exit $?
python -m pip install -e ${LFD_PKG_PATH}/lfd_smoother

echo ""
echo "Start installing APT dependencies..."
echo ""

apt-get update
cd ${LFD_PKG_PATH}/lfd_smoothing/docker/
grep -vE "^#" packages.list | xargs apt-get install -q -y --no-install-recommends || exit $?

echo ""
echo "Installation compelete."
echo ""
