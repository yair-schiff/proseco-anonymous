

import logging
import typing

import fsspec
import lightning
import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_info
from timm.scheduler import CosineLRScheduler


log = logging.getLogger(__name__)
warning_cache = WarningCache()


def fsspec_exists(filename):

  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):

  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):

  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):


  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch


    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:


  logger = logging.getLogger(name)
  logger.setLevel(level)


  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class CorrectorMetricEnabledModelCheckpoint(lightning.pytorch.callbacks.ModelCheckpoint):
    def __init__(self, start_monitor_step: int, **kwargs):
        super().__init__(**kwargs)
        self.start_monitor_step = start_monitor_step

    def _save_topk_checkpoint(
        self, trainer: "lightning.Trainer", monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        if self.save_top_k == 0:
            return
        if trainer.global_step < self.start_monitor_step:
          return


        if self.monitor is not None:
          if self.monitor not in monitor_candidates:
            m = (
              f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the"
              f" monitored key in the returned metrics: {list(monitor_candidates)}."
              f" HINT: Did you call `log({self.monitor!r}, value)` in the"
              f" `LightningModule`?"
            )
            if trainer.fit_loop.epoch_loop.val_loop._has_run:
              raise MisconfigurationException(m)
            warning_cache.warn(m)
          self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
          self._save_none_monitor_checkpoint(trainer, monitor_candidates)
