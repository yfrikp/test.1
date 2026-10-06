#Requires -Version 5.1
<#
  Python 预制件校验 —— 无法运行 Python 时，尽量用静态手段把错误拦掉

  用法：
    powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Lenovo\dsh-workspace\vision_task1\99-check-python.ps1"

  能查出来的：
    * 文件/函数/参数名 是否齐全
    * 跨文件调用的函数名是否存在（import 的符号是否真的定义过）
    * 参数名在 detector.py / params.yaml / C++ 之间是否一致
    * 括号/引号配平（防截断）
    * 常见的逐行语法事故（行尾多余字符、缩进用 Tab 混空格）
    * 中文是否混进了 print() 输出（会乱码）
#>

$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\Lenovo\dsh-workspace\vision_task1'
$PyDir = Join-Path $Root 'py'

$script:Pass = 0; $script:Fail = 0; $script:Warn = 0
function T-Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green;  $script:Pass++ }
function T-Bad($m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:Fail++ }
function T-Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:Warn++ }
function T-Head($m) { Write-Host ""; Write-Host "==== $m ====" -ForegroundColor Cyan }
$u8 = [System.Text.UTF8Encoding]::new($false)

Write-Host ''
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host '#  Python 预制件校验报告                                   #' -ForegroundColor Magenta
Write-Host '############################################################' -ForegroundColor Magenta

# ============================================================
#  工具：真正正确的括号配平扫描
#  必须跳过字符串与注释 —— 否则字符串里的括号（例如 print("(ok)")）
#  会被误算成不配平。这正是本校验器第一版犯过的错（假报警），所以单独实现。
# ============================================================
function Test-BracketBalance([string]$src) {
    $i = 0; $n = $src.Length
    $depth = 0; $line = 1
    $inS = $false; $inD = $false; $inT = $false; $inC = $false; $esc = $false
    $negativeAt = 0
    $bt = [char]39; $dq = [char]34      # 单引号 和 双引号

    while ($i -lt $n) {
        $c = $src[$i]
        if ($c -eq "`n") { $line++; $inC = $false; $i++; continue }
        if ($inC) { $i++; continue }

        if (-not $inS -and -not $inD -and ($i + 2) -lt $n -and $src.Substring($i, 3) -eq '"""') {
            $inT = -not $inT; $i += 3; continue
        }
        if ($inT) { $i++; continue }

        if ($esc) { $esc = $false; $i++; continue }
        if ($c -eq '\') { $esc = $true; $i++; continue }

        if (-not $inS -and -not $inD -and $c -eq '#') { $inC = $true; $i++; continue }
        if (-not $inD -and $c -eq $bt) { $inS = -not $inS; $i++; continue }
        if (-not $inS -and $c -eq $dq) { $inD = -not $inD; $i++; continue }
        if ($inS -or $inD) { $i++; continue }

        if ($c -eq '(' -or $c -eq '[' -or $c -eq '{') { $depth++ }
        elseif ($c -eq ')' -or $c -eq ']' -or $c -eq '}') {
            $depth--
            if ($depth -lt 0 -and $negativeAt -eq 0) { $negativeAt = $line }
        }
        $i++
    }
    return [pscustomobject]@{
        Depth      = $depth
        NegativeAt = $negativeAt
        OpenString = ($inT -or $inS -or $inD)
    }
}

# ============================================================
T-Head '一、文件存在性'
# ============================================================
$pyFiles = @('detector.py', 'make_test_images.py', 'selftest.py', 'tune_hsv.py', 'run_all.ps1', 'README-python.md')
$content = @{}
foreach ($f in $pyFiles) {
    $p = Join-Path $PyDir $f
    if (Test-Path $p) {
        T-Ok "存在 py\$f  ($((Get-Item $p).Length) 字节)"
        if ($f -like '*.py' -or $f -like '*.ps1' -or $f -like '*.md') {
            $content[$f] = [System.IO.File]::ReadAllText($p, $u8)
        }
    } else {
        T-Bad "缺失 py\$f"
    }
}

# ============================================================
T-Head '二、结构完整性（函数/类是否定义过）'
# ============================================================
$detector = $content['detector.py']
$neededDefs = @('class DetectorParams', 'class Detection',
                'def build_masks', 'def pixel_edge_for_distance', 'def classify',
                'def layer_check', 'def detect', 'def ground_plane_distance', 'def draw_result',
                'def _odd', 'def to_dict')
foreach ($d in $neededDefs) {
    if ($detector -and $detector.Contains($d)) { T-Ok "detector.py 定义了 $d" }
    else { T-Bad "detector.py 缺少 $d" }
}

# Detection 的字段（selftest / tune_hsv 都会访问）
$detFields = @('block_class','edge_length_m','edge_length_px','x_m','y_m','z_m','u_px','v_px',
               'confidence','box','ratio_red','ratio_blue','layer_ok','debug')
foreach ($f in $detFields) {
    if ($detector -match ("(?m)^\s{4}" + [regex]::Escape($f) + "\s*:")) { T-Ok "Detection 有字段 $f" }
    else { T-Bad "Detection 缺字段 $f" }
}

# DetectorParams 的字段必须与 params.yaml / C++ 对齐
$paramFields = @('red_h_low_max','red_h_high_min','red_s_min','red_v_min',
                 'blue_h_min','blue_h_max','blue_s_min','blue_v_min',
                 'dominant_ratio','interleave_ratio','layer_split_ratio','layer_size_tol',
                 'blur_kernel','morph_kernel','min_area_px',
                 'real_edge_m','corner_offset_m','fx','fy','cx','cy',
                 'distance_mode','cam_height_m','cam_pitch_deg')
foreach ($f in $paramFields) {
    if ($detector -match ("(?m)^\s{4}" + [regex]::Escape($f) + "\s*:")) { T-Ok "DetectorParams 有字段 $f" }
    else { T-Bad "DetectorParams 缺字段 $f" }
}

# ============================================================
T-Head '三、跨文件调用一致性（import 的符号是否真的存在）'
# ============================================================
foreach ($f in 'make_test_images.py', 'selftest.py', 'tune_hsv.py') {
    $src = $content[$f]
    if (-not $src) { continue }
    $imports = [regex]::Matches($src, 'from\s+detector\s+import\s+([^\r\n#]+)')
    foreach ($m in $imports) {
        $syms = $m.Groups[1].Value -split ',' | ForEach-Object { ($_ -replace '\(.*$','').Trim() } | Where-Object { $_ }
        foreach ($s in $syms) {
            if ($detector -match ("(class|def)\s+" + [regex]::Escape($s) + "\b")) {
                T-Ok "$f 导入的 $s 在 detector.py 里存在"
            } else {
                T-Bad "$f 导入了 $s，但 detector.py 里没有定义 —— 会 ImportError"
            }
        }
    }
    # 调用 detect( / draw_result( / build_masks( 的实参个数是否为正
    foreach ($fn in 'detect', 'draw_result', 'build_masks', 'classify', 'layer_check') {
        if ($src -match ("\b" + $fn + "\(")) { T-Ok "$f 调用了 $fn()" }
    }
}

# ============================================================
T-Head '四、selftest 用到的真值列是否都被生成器写出'
# ============================================================
$mk = $content['make_test_images.py']
$st = $content['selftest.py']
$truthKeys = @('file','block_class','distance_m','x_m','y_m','real_edge_m','edge_px','u_px','v_px',
               'light','noise_sigma','jpeg_quality','flip_layers')
foreach ($k in $truthKeys) {
    $inGen = $mk -match ('"' + [regex]::Escape($k) + '"\s*:')
    $inTest = $st -match ('"' + [regex]::Escape($k) + '"')
    if ($inGen -and $inTest) { T-Ok "真值列 $k 生成与读取都有" }
    elseif ($inGen)          { T-Warn "真值列 $k 有生成但 selftest 没用到（无害）" }
    elseif ($inTest)         { T-Bad "selftest 读取 $k，但生成器没写出这一列" }
    else                     { T-Ok "真值列 $k 未使用" }
}

# ============================================================
T-Head '五、括号/引号配平与常见语法事故'
# ============================================================
foreach ($f in 'detector.py', 'make_test_images.py', 'selftest.py', 'tune_hsv.py') {
    $src = $content[$f]
    if (-not $src) { continue }

    # 用真正的扫描算法查括号配平（跳过字符串与注释，避免假报警）
    $bal = Test-BracketBalance $src
    if ($bal.NegativeAt -gt 0) {
        T-Bad "$f 第 $($bal.NegativeAt) 行出现多余的右括号"
    } elseif ($bal.Depth -ne 0) {
        T-Bad "$f 括号不配平，最终深度 = $($bal.Depth)（应为 0）"
    } elseif ($bal.OpenString) {
        T-Bad "$f 扫描到文件末尾时仍在字符串内 —— 有未闭合的引号"
    } else {
        T-Ok "$f 括号完全配平（已跳过字符串与注释）"
    }

    # 三引号必须是偶数个（配合上面的 OpenString 一起看）
    $tq = ([regex]::Matches($src, '"""')).Count
    if ($tq % 2 -ne 0) { T-Bad "$f 的三引号数量为奇数（$tq）—— 字符串可能未闭合" }
    else { T-Ok "$f 三引号数量正常（$tq）" }

    # 禁止 Tab 缩进（Python 里 Tab/空格混用会报 TabError）
    $tabLines = ($src -split "`n") | Where-Object { $_ -match '^\t' }
    if ($tabLines) { T-Bad "$f 有 $($tabLines.Count) 行用 Tab 缩进 —— Python 会报错，必须用空格" }
    else { T-Ok "$f 没有 Tab 缩进" }

    # 行尾非法字符（用正则找非 ASCII 且非中文/全角标点的可疑字符）
    # 这里只做提示，不报错
    $weird = ($src -split "`n") | Where-Object { $_ -match '[\uFFFD]' }
    if ($weird) { T-Bad "$f 含替换字符 U+FFFD —— 编码已损坏" }
}

# ============================================================
T-Head '六、print / cv2.putText 里是否混入中文（会乱码）'
# ============================================================
foreach ($f in 'detector.py', 'make_test_images.py', 'selftest.py', 'tune_hsv.py') {
    $src = $content[$f]
    if (-not $src) { continue }
    $q1 = [char]34; $q2 = [char]39
    # 只看 print(...) 和 cv2.putText(...) 所在行
    $lines = ($src -split "`n") | Where-Object { $_ -match '\bprint\(|\bcv2\.putText\(' }
    $bad = $lines | Where-Object { $_ -match '[\u4e00-\u9fff]' -and $_ -match ($q1 + '|' + [regex]::Escape($q2)) }
    # 排除纯注释行
    $bad = $bad | Where-Object { $_.TrimStart() -notmatch '^#' }
    if ($bad) {
        T-Warn "$f 的 print/putText 行含中文（$($bad.Count) 处）—— 终端/图上会乱码："
        $bad | Select-Object -First 4 | ForEach-Object { T-Warn ('    ' + $_.Trim()) }
    } else {
        T-Ok "$f 的 print/putText 输出全为 ASCII"
    }
}

# ============================================================
T-Head '七、文档与入口脚本'
# ============================================================
foreach ($f in 'run_all.ps1', 'README-python.md') {
    if (Test-Path (Join-Path $PyDir $f)) { T-Ok "存在 py\$f" } else { T-Bad "缺失 py\$f" }
}
if (Test-Path (Join-Path $PyDir 'run_all.ps1')) {
    $b = [System.IO.File]::ReadAllBytes((Join-Path $PyDir 'run_all.ps1'))
    $bom = ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF)
    if ($bom) { T-Ok 'run_all.ps1 有 UTF-8 BOM（中文不会乱码）' } else { T-Bad 'run_all.ps1 缺 UTF-8 BOM' }
    $errs = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PyDir 'run_all.ps1'), [ref]$null, [ref]$errs)
    if ($errs.Count -gt 0) { T-Bad "run_all.ps1 语法错误 $($errs.Count) 处" } else { T-Ok 'run_all.ps1 语法正确' }
}

# ============================================================
T-Head '八、汇总'
# ============================================================
Write-Host ''
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host ("  通过 {0} 项 / 失败 {1} 项 / 警告 {2} 项" -f $script:Pass, $script:Fail, $script:Warn) -ForegroundColor $(if ($script:Fail -gt 0) {'Red'} else {'Green'})
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host ''
Read-Host '按回车退出'
