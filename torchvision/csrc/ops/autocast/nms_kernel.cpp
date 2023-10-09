#include "../nms.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  printf("in gpu autocast\n");
  return nms(
      at::autocast::cached_cast(at::kFloat, dets),
      at::autocast::cached_cast(at::kFloat, scores),
      iou_threshold);
}

at::Tensor nms_autocast_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  printf("in nms_autocast_cpu\n");
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(c10::DispatchKey::AutocastCPU);
  return nms(
      at::autocast::cached_cast(at::kFloat, dets, c10::DeviceType::CPU),
      at::autocast::cached_cast(at::kFloat, scores, c10::DeviceType::CPU),
      iou_threshold);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_autocast));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_autocast_cpu));
}

} // namespace ops
} // namespace vision
