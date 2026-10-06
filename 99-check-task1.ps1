#Requires -Version 5.1
<#
  题目1 预制件校验 —— 在交给 Ubuntu 编译之前，先把能静态查出来的错误拦掉

  用法：
    powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Lenovo\dsh-workspace\vision_task1\99-check-task1.ps1"

  为什么要这么严：
    ROS2 的编译错误信息很长且经常指向生成的文件，新手很难定位。
    这里把「消息字段名写错、CMakeLists 缺依赖、YAML 缺参数、字符串里混中文」
    这类问题在 Windows 侧就查出来。
#>

$ErrorActionPreference = 'Continue'
$Root    = 'C:\Users\Lenovo\dsh-workspace\vision_task1'
$MsgPkg  = Join-Path $Root 'src\vision_task1_interfaces'
$NodePkg = Join-Path $Root 'src\vision_task1'

$script:Pass = 0; $script:Fail = 0; $script:Warn = 0
function T-Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green;  $script:Pass++ }
function T-Bad($m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:Fail++ }
function T-Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:Warn++ }
function T-Head($m) { Write-Host ""; Write-Host "==== $m ====" -ForegroundColor Cyan }
$Q = [char]34

Write-Host ''
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host '#  题目1 预制件校验报告                                     #' -ForegroundColor Magenta
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host ("  时间：{0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))

# ============================================================
T-Head '一、文件结构'
# ============================================================
$need = @(
  'src\vision_task1_interfaces\package.xml',
  'src\vision_task1_interfaces\CMakeLists.txt',
  'src\vision_task1_interfaces\msg\BlockInfo.msg',
  'src\vision_task1\package.xml',
  'src\vision_task1\CMakeLists.txt',
  'src\vision_task1\src\color_detector_node.cpp',
  'src\vision_task1\config\params.yaml'
)
foreach ($f in $need) {
  $p = Join-Path $Root $f
  if (Test-Path $p) { T-Ok "存在 $f" } else { T-Bad "缺失 $f" }
}

# ============================================================
T-Head '二、BlockInfo.msg 字段完整性（接口定错则全部返工）'
# ============================================================
$msgPath = Join-Path $MsgPkg 'msg\BlockInfo.msg'
$msgFields = @()
if (Test-Path $msgPath) {
  $raw = [System.IO.File]::ReadAllText($msgPath, [System.Text.UTF8Encoding]::new($false))
  # 剥掉 # 注释与空行，剩下 type name 形式
  $lines = ($raw -split "`n") | ForEach-Object { ($_ -replace '#.*$','').Trim() } | Where-Object { $_ }
  foreach ($l in $lines) {
    $m = [regex]::Match($l, '^([\w/]+)\s+(\w+)$')
    if ($m.Success) { $msgFields += @{ type = $m.Groups[1].Value; name = $m.Groups[2].Value } }
    else { T-Warn "无法解析这一行：$l" }
  }
  T-Ok "解析出 $($msgFields.Count) 个字段：$(($msgFields | ForEach-Object { $_.name }) -join ', ')"
} else {
  T-Bad 'BlockInfo.msg 不存在'
}

# 题目要求的三个信息必须都在
$requiredFields = @(
  @{ n='block_class';    why='题目要求输出「种类」' },
  @{ n='edge_length_m';  why='题目要求输出「长度」' },
  @{ n='x_m';            why='题目要求输出「距离摄像头中心点的坐标」的 X' },
  @{ n='y_m';            why='题目要求输出「距离摄像头中心点的坐标」的 Y' },
  @{ n='z_m';            why='题目要求输出「距离摄像头中心点的坐标」的 Z（距离）' },
  @{ n='header';         why='时间戳用于验证话题频率，frame_id 用于将来接 TF 做偏移补偿' }
)
foreach ($rf in $requiredFields) {
  if ($msgFields | Where-Object { $_.name -eq $rf.n }) { T-Ok "字段 $($rf.n) 存在（$($rf.why)）" }
  else { T-Bad "字段 $($rf.n) 缺失 —— $($rf.why)" }
}
# 辅助字段（有更好）
foreach ($af in 'u_px','v_px','edge_length_px','confidence') {
  if ($msgFields | Where-Object { $_.name -eq $af }) { T-Ok "辅助字段 $af 存在" }
  else { T-Warn "辅助字段 $af 缺失（不影响评分，但调试时有用）" }
}

# ============================================================
T-Head '三、节点代码是否用到了全部字段（消息与代码一致性）'
# ============================================================
$nodePath = Join-Path $NodePkg 'src\color_detector_node.cpp'
$node = ''
if (Test-Path $nodePath) {
  $node = [System.IO.File]::ReadAllText($nodePath, [System.Text.UTF8Encoding]::new($false))
  $bytes = [System.IO.File]::ReadAllBytes($nodePath)
  $cr = 0; foreach ($byte in $bytes) { if ($byte -eq 13) { $cr++ } }
  if ($cr -gt 0) { T-Warn "color_detector_node.cpp 含 CRLF（build 脚本会转 LF）" } else { T-Ok 'color_detector_node.cpp 为 LF 换行' }

  # 花括号配平（防截断）
  $ob = ([regex]::Matches($node, '\{')).Count
  $cb = ([regex]::Matches($node, '\}')).Count
  if ($ob -eq $cb) { T-Ok "花括号配平（$ob 对）" } else { T-Bad "花括号不配平：$ob vs $cb —— 代码可能被截断" }

  foreach ($f in ($msgFields | ForEach-Object { $_.name })) {
    if ($node -match ("out\.$f\b")) { T-Ok "代码里给 out.$f 赋了值" }
    else { T-Bad "代码里没有给 out.$f 赋值 —— 发布出去会是空值" }
  }
}

# ============================================================
T-Head '四、CMakeLists 与 package.xml'
# ============================================================
foreach ($pkg in @(@{n='vision_task1_interfaces'; d=$MsgPkg; deps=@('rosidl_default_generators','std_msgs','ament_cmake')},
                   @{n='vision_task1';            d=$NodePkg; deps=@('rclcpp','sensor_msgs','cv_bridge','OpenCV','vision_task1_interfaces')})) {
  $px = Join-Path $pkg.d 'package.xml'
  if (Test-Path $px) {
    try {
      [xml]$x = Get-Content -LiteralPath $px -Raw -Encoding UTF8
      if ($x.package.name -eq $pkg.n) { T-Ok "$($pkg.n)/package.xml 合法，name 正确" }
      else { T-Bad "$($pkg.n)/package.xml 里 name 是 '$($x.package.name)'，应为 $($pkg.n)" }
      if ($x.package.export.build_type -eq 'ament_cmake') { T-Ok "$($pkg.n) build_type=ament_cmake" } else { T-Bad "$($pkg.n) build_type 错误" }
    } catch { T-Bad "$($pkg.n)/package.xml 不是合法 XML：$($_.Exception.Message)" }
  }
  $cl = Join-Path $pkg.d 'CMakeLists.txt'
  if (Test-Path $cl) {
    $t = [System.IO.File]::ReadAllText($cl, [System.Text.UTF8Encoding]::new($false))
    $missing = @()
    foreach ($d in $pkg.deps) {
      if ($t -notmatch ("find_package\(\s*" + [regex]::Escape($d))) { $missing += $d }
    }
    if ($missing.Count -eq 0) { T-Ok "$($pkg.n)/CMakeLists.txt 的 find_package 齐全" }
    else { T-Bad "$($pkg.n)/CMakeLists.txt 缺少 find_package：$($missing -join ', ')" }
    $o = ([regex]::Matches($t, '\(')).Count; $c = ([regex]::Matches($t, '\)')).Count
    if ($o -eq $c) { T-Ok "$($pkg.n)/CMakeLists.txt 括号配平（$o 对）" } else { T-Bad "$($pkg.n)/CMakeLists.txt 括号不配平 $o vs $c" }
    if ($t -match 'ament_package\(\)') { T-Ok "$($pkg.n) 有 ament_package()" } else { T-Bad "$($pkg.n) 缺 ament_package()" }
  }
}
# 消息包特有：必须声明 rosidl_generate_interfaces
$msgCml = Join-Path $MsgPkg 'CMakeLists.txt'
if (Test-Path $msgCml) {
  $t = [System.IO.File]::ReadAllText($msgCml, [System.Text.UTF8Encoding]::new($false))
  if ($t -match 'rosidl_generate_interfaces') { T-Ok '消息包调用了 rosidl_generate_interfaces' } else { T-Bad '消息包缺 rosidl_generate_interfaces' }
  if ($t -match 'BlockInfo\.msg') { T-Ok '消息包列出了 BlockInfo.msg' } else { T-Bad '消息包没有列出 BlockInfo.msg' }
}
# 节点包特有：必须链接 OpenCV
$nodeCml = Join-Path $NodePkg 'CMakeLists.txt'
if (Test-Path $nodeCml) {
  $t = [System.IO.File]::ReadAllText($nodeCml, [System.Text.UTF8Encoding]::new($false))
  if ($t -match 'target_link_libraries\([^)]*OpenCV_LIBS') { T-Ok '节点包链接了 OpenCV_LIBS' } else { T-Bad '节点包没有链接 OpenCV_LIBS —— 会报 undefined reference' }
  if ($t -match 'install\(TARGETS') { T-Ok '节点包有 install(TARGETS ...)' } else { T-Bad '节点包缺 install(TARGETS ...) —— ros2 run 找不到程序' }
}

# ============================================================
T-Head '五、params.yaml 与代码里的参数名是否一致'
# ============================================================
$yamlPath = Join-Path $NodePkg 'config\params.yaml'
if ((Test-Path $yamlPath) -and $node) {
  $yaml = [System.IO.File]::ReadAllText($yamlPath, [System.Text.UTF8Encoding]::new($false))
  # 从代码里抓 declare_parameter("名字"
  $declared = [regex]::Matches($node, 'declare_parameter\(\s*' + $Q + '(\w+)' + $Q) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
  $inYaml   = [regex]::Matches($yaml, '^\s{4}(\w+):', 'Multiline') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

  $notInYaml = $declared | Where-Object { $_ -notin $inYaml }
  $notInCode = $inYaml   | Where-Object { $_ -notin $declared }
  if ($notInYaml.Count -eq 0) { T-Ok "代码里声明的 $($declared.Count) 个参数全部出现在 params.yaml" }
  else { T-Warn "这些参数在代码里有、YAML 里没有（会用代码默认值）：$($notInYaml -join ', ')" }
  if ($notInCode.Count -eq 0) { T-Ok 'params.yaml 里没有多余的参数' }
  else { T-Bad "params.yaml 里有代码未声明的参数（ROS2 会直接报错拒绝启动）：$($notInCode -join ', ')" }
}

# ============================================================
T-Head '六、字符串字面量里是否混入中文（会导致终端输出乱码）'
# ============================================================
if ($node) {
  $codeOnly = ($node -split "`n" | ForEach-Object { $_ -replace '(^|\s)//.*$', '$1' }) -join "`n"
  $hits = ($codeOnly -split "`n") | Where-Object { $_ -match ($Q + '[^' + $Q + ']*[\u4e00-\u9fff][^' + $Q + ']*' + $Q) }
  if ($hits) {
    T-Bad "字符串字面量里含中文（$($hits.Count) 处）—— 日志会乱码："
    $hits | Select-Object -First 5 | ForEach-Object { T-Bad ('    ' + $_.Trim()) }
  } else {
    T-Ok '字符串字面量里无中文（终端日志不会乱码）'
  }
}

# ============================================================
T-Head '七、加分项与 60Hz 实现是否在位'
# ============================================================
if ($node) {
  $checks = @{
    '偏移补偿参数 corner_offset_m 已声明' = 'declare_parameter\(\s*' + $Q + 'corner_offset_m'
    '偏移补偿逻辑已实现'                   = 'corner_offset_m_\s*>\s*1e-9|x\s*-=\s*t'
    '物块真实边长参数 real_edge_m 已声明'  = 'declare_parameter\(\s*' + $Q + 'real_edge_m'
    '60Hz 定时器已创建'                    = 'create_wall_timer\(\s*[^)]*republish_hz|republish_hz_'
    '频率自检已实现（验收证据）'            = 'report_rate'
    '未达 60Hz 时有告警提示'               = 'Below 60Hz'
    '英文日志（避免 GBK 终端乱码）'         = 'Static image mode|Topic mode'
    '三分类判定齐全（red/blue/red_blue）'   = '"red_blue"'
    '红蓝相间有几何交叉验证（分层）'         = 'layer_check'
  }
  foreach ($k in $checks.Keys | Sort-Object) {
    if ($node -match $checks[$k]) { T-Ok $k } else { T-Bad "$k —— 未找到" }
  }
}

# ============================================================
T-Head '八、汇总'
# ============================================================
Write-Host ''
Write-Host '  即将交付的题目1 仓库结构：'
Write-Host '    vision_task1\' -ForegroundColor White
Write-Host '      src\vision_task1_interfaces\   msg\BlockInfo.msg        <- 接口定义'
Write-Host '      src\vision_task1\              src\color_detector_node.cpp  <- 算法实现'
Write-Host '                                     config\params.yaml      <- 你要调的参数'
Write-Host ''
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host ("  通过 {0} 项 / 失败 {1} 项 / 警告 {2} 项" -f $script:Pass, $script:Fail, $script:Warn) -ForegroundColor $(if ($script:Fail -gt 0) {'Red'} else {'Green'})
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host ''
Read-Host '按回车退出'
