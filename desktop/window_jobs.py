"""主窗口的作业命令：作业管理器的动作，复用现有的后台求解与结果生效路径。"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QMessageBox

from .job_manager import CreateJobDialog, Job, JobManagerDialog, _now, describe_result
from .object_dialogs import next_name


class WindowJobsMixin:
    """作业管理。见 job_manager.py。"""

    @property
    def jobs(self) -> list[Job]:
        if not hasattr(self, "_jobs"):
            self._jobs = []
        return self._jobs

    def open_job_manager(self) -> None:
        dialog = getattr(self, "_job_dialog", None)
        if dialog is None:
            self._job_dialog = dialog = JobManagerDialog(self)
        dialog.refresh()
        dialog.show()
        dialog.raise_()

    def _refresh_job_dialog(self) -> None:
        dialog = getattr(self, "_job_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.refresh()

    def job_action(self, action: str, job: Job | None) -> None:
        if action == "create":
            dialog = CreateJobDialog(next_name("Job", {j.name for j in self.jobs}), self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                name, kind, analysis = dialog.values()
                if name and name not in {j.name for j in self.jobs}:
                    self.jobs.append(Job(name, kind, analysis))
            return
        if job is None:
            return
        if action == "check":
            errors = self.session.validation_errors()
            if job.kind == "steps" and not self.session.model.get("steps"):
                errors = [*errors, "模型里没有声明分析步：请先在分析步管理器创建，或改用独立工况作业"]
            job.status = "数据检查未通过" if errors else "数据检查通过"
            job.messages = [f"[{_now()}] 数据检查"] + (errors or ["模型完整，可以提交。"])
        elif action == "inp":
            self._write_job_inp(job)
        elif action == "submit":
            self.submit_job(job)
        elif action == "results":
            self._show_job_results(job)
        elif action == "kill":
            self.stop_task()
        elif action == "delete":
            if job.status != "运行中":
                self.jobs.remove(job)

    def _write_job_inp(self, job: Job) -> None:
        from inp_writer import write_inp
        dialog = getattr(self, "_job_dialog", None)
        path = dialog.ask_inp_path(job) if dialog is not None else ""
        if not path:
            return
        errors = self.session.validation_errors()
        if errors:
            job.messages.append(f"[{_now()}] 写 inp 失败：模型未通过校验（{errors[0]}）")
            return
        written = write_inp(self.session.model, path=path)
        job.messages.append(f"[{_now()}] 已写出 Abaqus 输入文件：{written}")

    def submit_job(self, job: Job) -> None:
        """后台提交。独立工况与分析步各走原有的求解与结果生效路径。"""
        if self.runner.busy:
            return
        if job.kind == "steps" and not self.session.model.get("steps"):
            job.status = "未提交"
            job.messages = [f"[{_now()}] 模型里没有声明分析步，无法提交"]
            return
        job.status, job.submitted, job.finished = "运行中", _now(), ""
        job.fingerprint = self.session._model_key()
        job.messages = [f"[{_now()}] 已提交：{job.label}"]
        self._on_busy(True)
        self.statusBar().showMessage(f"作业 {job.name} 运行中…")
        if job.kind == "steps":
            work, finish = self.session.solve_steps, self._steps_solved
        else:
            def work():
                return self.session.solve_model(analysis=job.analysis, **self.analysis_options)
            finish = self._solved
        if not self.runner.submit(work, on_done=lambda result: self._job_done(job, result, finish),
                                  on_failed=lambda kind, message: self._job_failed(job, kind,
                                                                                    message)):
            self._on_busy(False)
            job.status = "未提交"
        self._refresh_job_dialog()

    def _job_done(self, job: Job, result, finish) -> None:
        job.finished = _now()
        job.status = "完成" if result.ok else "失败"
        if result.ok:
            # 会话一次只端一份结果：记下它属于哪个作业，旧作业的「结果」才不会张冠李戴
            self._result_job = job
        job.messages += describe_result(job, result)
        finish(result)
        self._refresh_job_dialog()

    def _job_failed(self, job: Job, kind: str, message: str) -> None:
        job.finished = _now()
        job.status = "已终止" if kind == "TaskCancelled" else "失败"
        job.messages.append(f"[{_now()}] {kind}: {message[:400]}")
        self._solve_failed(kind, message)
        self._refresh_job_dialog()

    def _show_job_results(self, job: Job) -> None:
        if job.status != "完成":
            self.statusBar().showMessage(f"作业 {job.name} 还没有完成，没有结果可看。", 5000)
            return
        if (job.fingerprint != self.session._model_key() or self.session.solution is None
                or getattr(self, "_result_job", None) is not job):
            QMessageBox.information(
                self, "结果对应旧模型",
                f"作业 {job.name} 提交之后模型已经修改（或有别的作业覆盖了结果），"
                "当前看到的不是这次作业的结果。请重新提交该作业。")
            return
        self.set_mode("变形")
        self.results_dock.show()
