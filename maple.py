import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from features import polynomials


def project(text, variable, remaining, *, executable="maple", timeout=900):
    executable = shutil.which(str(executable))
    if executable is None:
        raise FileNotFoundError("Maple executable not found; supply --maple /path/to/maple")
    executable = str(Path(executable).resolve())
    if timeout <= 0:
        raise TimeoutError("UP prediction exceeded its time budget")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", v) for v in [variable, *remaining]):
        raise ValueError("Unsupported Maple variable name")
    polys = ",".join(str(p.as_expr()).replace("**", "^") for p in polynomials(text))
    package = str(Path(__file__).with_name("projection.mpl").resolve())
    with tempfile.TemporaryDirectory(prefix="grl-svo-maple-") as directory:
        directory = Path(directory)
        result = directory / "result.txt"
        script = directory / "job.mpl"
        script.write_text(
            "interface(prettyprint=0):\nkernelopts(printbytes=false):\n"
            f"read({json.dumps(package)}):\nwith(projection):\n"
            f"F := {{{polys}}}:\n"
            f"P := PCAD_McCallumProj(F,{variable},[{','.join(remaining)}]):\n"
            "answer := PCAD_SetFactors(P):\n"
            "answer := map(x->expand(denom(x)*x),answer):\n"
            f"writestat({json.dumps(str(result))}, answer):\nquit;\n")
        with (directory / "output.log").open("w+") as log:
            process = subprocess.Popen([executable, "-q", str(script)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       cwd=directory, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError("Maple projection exceeded its time budget") from None
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            log.seek(0)
            message = log.read()
        if process.returncode or not result.exists() or "Error," in message:
            raise RuntimeError(f"Maple failed: {message[-3000:]}")
        value = "".join(result.read_text().splitlines()).strip()
        if value in ("answer", "projs_fac", "FAIL"):
            raise RuntimeError(f"Maple returned an unevaluated result: {value}")
        return value
