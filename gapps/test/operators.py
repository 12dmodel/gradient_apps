import os
import time
import unittest
import skimage.io

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

import gapps.utils as utils
import gapps.functions.operators as ops

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

# ----------- cpu/gpu test calls --------------------------------------------
def test_bilateral_layer_cpu():
  _test_bilateral_layer_(gpu=False)

def test_bilateral_layer_gpu():
  _test_bilateral_layer_(gpu=True)

def test_histogram_cpu():
  _test_histogram(False)

def test_histogram_gpu():
  _test_histogram(True)

def test_soft_histogram_cpu():
  _test_soft_histogram(False)

def test_soft_histogram_gpu():
  _test_soft_histogram(True)
# ---------------------------------------------------------------------------


def _test_bilateral_layer_(gpu=False):
  bs = 1;
  ci = 3
  co = 1
  kh = 3
  kw = 3
  kd = 3

  sx, sy, sz = 4, 4, 8

  image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))
  guide = skimage.io.imread(os.path.join(data_dir, "gray.png"))

  skimage.io.imsave(os.path.join(out_dir, "bilateral_layer_input.png"), image)
  sz = 256
  image = image[:sz, :sz, :]
  guide = guide[:sz, :sz]

  h, w = guide.shape
  image = np.expand_dims(image.transpose([2, 0 , 1])/255.0, 0).astype(np.float32)
  guide = np.expand_dims(guide/255.0, 0).astype(np.float32)

  image = Variable(th.from_numpy(image), requires_grad=False)
  guide = Variable(th.from_numpy(guide), requires_grad=False)
  kernels = Variable(th.randn(co, ci, kd, kh, kw), requires_grad=True)

  conv = th.nn.Conv2d(ci, co, [kh*sy, kw*sx])

  if gpu:
    image = image.cuda()
    guide = guide.cuda()
    kernels = kernels.cuda()
    conv.cuda()

  print "profiling"
  with profiler.profile() as prof:
    output = ops.BilateralLayer.apply(
        image, guide, kernels, sx, sy, sz)
    output2 = conv(image)
    loss = output.sum()
    loss.backward()

  print prof

  print "testing dimensions"
  assert output.shape[0] == bs
  assert output.shape[1] == co
  assert output.shape[2] == h
  assert output.shape[3] == w

  print "testing forward"
  for i, o in enumerate([output, output2]):
    mini, maxi = o.min(), o.max()
    o -= mini
    o /= (maxi-mini)

    o = o.data[0].cpu().numpy()
    o = np.clip(np.transpose(o, [1, 2, 0]), 0, 1)
    o = np.squeeze(o)
    skimage.io.imsave(
        os.path.join(out_dir, "bilateral_layer_{}.png".format(i)), o)

  # print "testing gradient"
  # gradcheck(
  #     ops.BilateralLayer.apply,
  #     (image, guide, kernels, sx, sy, sz),
  #     eps=1e-4, atol=5e-2, rtol=5e-4,
  #      raise_exception=True)


# def test_ahd_demosaick():
#   image = skimage.io.imread(
#       os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
#   h, w, _ = image.shape
#   mosaick = utils.make_mosaick(image)
#   skimage.io.imsave(
#       os.path.join(out_dir, "ahd_mosaick.png"), mosaick)
#
#   mosaick = Variable(th.from_numpy(mosaick), requires_grad=False)
#   print "profiling"
#   with profiler.profile() as prof:
#     output = ops.AHDDemosaick.apply(mosaick)
#     # loss = output.sum()
#     # loss.backward()
#
#   print prof
#
#   assert output.shape[0] == 3
#   assert output.shape[1] == h
#   assert output.shape[2] == w
#
#   output = output.data.numpy()
#   output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
#   output = np.squeeze(output)
#   skimage.io.imsave(
#       os.path.join(out_dir, "ahd_demosaicked.png"), output)


def _test_histogram(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "gray.png")).astype(np.float32)/255.0
  image = image[:64, :64]
  nbins = 8

  image = Variable(th.from_numpy(image), requires_grad=True)
  if gpu:
    image = image.cuda()

  print "profiling"
  with profiler.profile() as prof:
    output = ops.Histogram.apply(image, nbins)
    loss = output.sum()
    loss.backward()
  print prof

  print "checking gradients"
  image = image[:32, :32]
  gradcheck(ops.Histogram.apply,
      (image, nbins), eps=1e-4, atol=5e-2, rtol=5e-4,
      raise_exception=True)


def _test_soft_histogram(gpu=False):
  image = skimage.io.imread(os.path.join(data_dir, "gray.png")).astype(np.float32)/255.0
  image = image[:64, :64]
  nbins = 8

  image = Variable(th.from_numpy(image), requires_grad=True)
  if gpu:
    image = image.cuda()

  print "profiling"
  with profiler.profile() as prof:
    output = ops.SoftHistogram.apply(image, nbins)
    loss = output.sum()
    loss.backward()
  print prof
