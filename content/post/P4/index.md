+++
date = '2026-11-09T20:16:12-05:00'
draft = false
title = '北航计组 P4 上机经验'
tags = ["verilog", "P4"]
categories = ["CO"]
math = true
+++

# 思路分享

## NIBSRT

这里新建的模块可以直接放在 `always` 块里面，然后使用冒泡排序或者其他的方法即可完成排序，然后进行写入即可，但是这一道题我 A 了1h也没有做出来，原因是没有注意到题目需要的是 `rt` 中 1 的数量，而不是 `GRF[rt]` 中的数量

## BLEZAL

这道题没有什么特别好说的，在 `NPC` 里面加多上一个新路，然后再传入一个 `GRF[rs]` 即可

## LWSEL

同上，这里直接在 `mips.v` 新增一个模块用来产出要写入的地址即可

题面记录：[BUAA-CO-2026](https://github.com/Lunemere/BUAA-CO-2026)