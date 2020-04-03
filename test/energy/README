# ecint 工作流基本文件介绍 -- 以 energy 为例

## h2o.xyz

> 结构文件，一般通过 ase 来生成

若xyz的注释行(第二行)是空的，那么就没有 cell 与 pbc 的信息，那么在读取结构文件后还需要进行 set_cell 与 set_pbc(可选) 的设置

`h2o.xyz` 已包括 cell 与 pbc，因此直接读入结构文件即可

## energy.json

> 主要以此文件为模板生成 cp2k 输入文件 aiida.inp

energy 计算的基本设置，但是不包括结构以及 kind 的信息

此文件可由用户定义，也可直接用预定义好的文件 (ecint.workflow.units)

这个例子中其实并未使用此文件夹下的 `energy.json`，而是使用预定义好的 `energy.json`

## kind_section.yaml

> 设置cp2k输入文件的 kind 部分，此文件不是必须的，默认基组用 `DZVP-MOLOPT-SR-GTH`，默认赝势用 `GTH-BLYP`

此文件可由用户定义，也可使用默认值

## machine.json

> 设置提交服务器的相关信息，主要包括提交脚本设置与 `code@computer` 设置

目前仅开放了 LSF 系统的 `#BSUB -n/-W/-q/-R` 设置

## energy.py & run.py

> energy.py 为定义好的完整的 energy work chain class，run.py 为提交工作流的脚本

warning: 不能用 \_\_name\_\_=='\_\_main\_\_' 的方法把两个文件合并成一个，即工作流的定义与使用不能放在同一个文件中，否则会有错误

运行工作流结束后，会自动把结果写入到 `results.txt` 中

## results.txt

> energy 工作流运行好后自动输出的结果文件

