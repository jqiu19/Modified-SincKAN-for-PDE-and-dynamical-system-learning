$script = "/mnt/c/Users/Qiu Jingwei/Documents/New project/jaxpi/examples/ldc/run_sinc_re3200.sh"
$stdout = "C:\Users\Qiu Jingwei\Documents\New project\jaxpi\examples\ldc\sinc_re3200_run.log"
$stderr = "C:\Users\Qiu Jingwei\Documents\New project\jaxpi\examples\ldc\sinc_re3200_run.err"

Remove-Item -LiteralPath $stdout -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderr -ErrorAction SilentlyContinue

Start-Process `
  -FilePath "wsl.exe" `
  -ArgumentList @("-d", "Ubuntu-22.04", "bash", $script) `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr
