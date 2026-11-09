+++
date = '2026-11-09T00:42:52-05:00'
draft = true
title = '波的一些公式 Part 2'
tags = ["Homework", "知识点总结"]
categories = ["基础物理学"]
math = true
+++

辐射通量就是 $\bar{S}$ ，辐射功率是通过一整块面积的总辐射通量，若是球面：
$$
P = \bar{S} \cdot 4 \pi R^2
$$
辐射压强，光束照到物体表面实际是光子发生弹性碰撞，那么辐射压强：
$$
p = \frac{2 \bar{S}}{c}
$$

> 这是在全部反射的情况下

平均辐射强度就是坡印廷矢量平均值，若光的功率是 $P$ 那么平均辐射强度是，这是平面的电磁波
$$
\bar{S} = \frac{P}{A}
$$
电场强度（标量为电场幅值）和平均坡印廷矢量的关系是：
$$
\bar{S} = \frac{1}{2} \epsilon_0 c E_0^2
$$
磁感应强度和电场强度的关系（真空中）：
$$
B_0 = \frac{E_0}{c}
$$
两列相干波在空间某点相遇时的总相位差公式：

$$
\Delta\varphi = (\varphi_{10} - \varphi_{20}) - \frac{2\pi}{\lambda}(r_1 - r_2)
$$

> $r_1$ 与 $r_2$ 是该点到两个波源的距离，然后 $\varphi_{10}, \varphi_{20}$ 代表是初相位差

两列波合成波的强度公式是，频率相同时相干波：
$$
I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos(\Delta\varphi)
$$

其中 $\Delta \varphi$ 是相位差。

计算两个波的干涉是增强还是减弱，前者为加强，后者为减弱：

$$
\Delta \varphi_1 = \pm 2k\pi
$$

$$
\Delta \varphi_2 = \pm(2k+1)\pi
$$

波源在传播方向上到达某点引起的分振动表达式
$$
y_{i} = A \cos\left[\phi_i(t) - \frac{2\pi}{\lambda} r_i\right]
$$

> $r_i$ 是波源在某点的距离

要求某点的振动表达式，就把x的值代入进去

在波密介质会发生反射，反射会发生 $\pi$ 的相位图片，需要在 $\cos$ 里面加上一个 $\pi$ 

> 在波传播的过程中，向前走的距离是要取负号的 $y_2(x, t) = A \cos\left[2\pi\nu t - \frac{2\pi}{\lambda}(x_0 - x)\right] $

合成波就只要把两个波的表达式相加，就会形成驻波，eg.

$$
y = -2A \cos\left(\frac{2\pi x}{\lambda}\right) \sin(2\pi\nu t)
$$
要求出波节，就令前面这个x等于0.波腹让前面的x绝对值等于1即可

$$
\nu' = \frac{u + v_O}{u} \nu_0
$$

这是观察者向波源移动的多普勒定律的情况，如果是反方向在下方改变符号

$$
\nu_1 = \frac{u}{u + v_{\text{S}}} \nu_0
$$

这是波源向观察者移动，如果是反方向在下方改变符号

$$
\Delta \nu = \nu_2 - \nu_1
$$

拍频是两列波的频率差
