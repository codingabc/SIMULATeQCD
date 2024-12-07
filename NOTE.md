# 术语
__HISQ__: highly improved staggered quark fermions
__rhmc__: the rational hybrid Monte Carlo algorithm

# 背景知识

在矩阵运算中，“dagger”通常指的是“厄米共轭（Hermitian conjugate）”，也称为“ dagger 操作”或“ dagger 运算”。
对于一个矩阵 \(A\) ，它的厄米共轭 \(A^\dagger\) 定义为 \(A^\dagger = (A^*)^T\) ，即先对 \(A\) 取复共轭（元素的实部不变，虚部取相反数），然后再取转置。


在数学中，“\(\partial\)”符号通常表示偏导数。
例如，\(\frac{\partial H}{\partial p_i}\) 表示哈密顿函数 \(H\) 关于广义动量 \(p_i\) 的偏导数，意味着在计算导数时，把除了 \(p_i\) 之外的其他变量都视为常数。


GeV是吉电子伏特（Giga - electron - volt）的缩写，是能量单位。

- **电子伏特（eV）的定义**
  电子伏特是一个能量单位，它的定义是一个电子（电量为\(1.6\times10^{-19}\)库仑）经过\(1\)伏特的电位差加速后所获得的动能。根据能量计算公式\(E = qU\)（其中\(E\)是能量，\(q\)是电荷量，\(U\)是电位差），对于电子伏特，\(q = e=1.6\times10^{-19}\text{C}\)，\(U = 1\text{V}\)，所以\(1\)电子伏特\(=1.6\times10^{-19}\text{J}\)（焦耳）。

- **吉电子伏特（GeV）与电子伏特（eV）的换算关系**
  “吉（Giga）”是一个数量级单位，表示\(10^{9}\)。所以\(1\text{GeV}=10^{9}\text{eV}\)，换算成焦耳就是\(1\text{GeV}=1.6\times10^{-10}\text{J}\)。

- **应用领域**
  在粒子物理学中，GeV是一个常用的能量单位。因为在研究亚原子粒子，如质子、中子、电子、各种介子等的能量时，这些粒子的能量通常很高。例如，欧洲核子研究中心（CERN）的大型强子对撞机（LHC）可以将质子加速到能量高达数TeV（\(1\text{TeV}=10^{12}\text{eV}\)）的水平，在一些早期的粒子加速器或者对较低能量的粒子过程研究中，GeV这个单位就很合适用于描述粒子的能量。

**Leapfrog 算法**
- **定义和基本原理**
   - Leapfrog算法是一种数值积分方法，主要用于求解常微分方程（ODE）。它是一种二阶精度的方法，特别适用于求解哈密顿系统（在物理学中，哈密顿系统描述了一个物理系统的能量和动力学）。
   - 对于一个二阶常微分方程\(\ddot{x}=f(x)\)（其中\(x\)是位置，\(\ddot{x}\)是加速度，\(f(x)\)是力相关的函数），可以将其改写为一阶常微分方程组。设\(v = \dot{x}\)（速度），那么方程组为\(\dot{x}=v\)和\(\dot{v}=f(x)\)。
   - Leapfrog算法的基本思想是交错地更新位置和速度。它通过在时间步长\(\Delta t\)上进行离散化来近似求解运动方程。位置\(x\)和速度\(v\)的更新公式如下：
     - 首先更新速度的半步长：\(v_{n + \frac{1}{2}}=v_{n}+\frac{1}{2}\Delta t f(x_{n})\)
     - 然后更新位置：\(x_{n+1}=x_{n}+\Delta t v_{n+\frac{1}{2}}\)
     - 最后更新速度的另一半步长：\(v_{n + 1}=v_{n+\frac{1}{2}}+\frac{1}{2}\Delta t f(x_{n + 1})\)

- **优点**
   - **高精度**：它是二阶精度的方法，相比于一些一阶方法（如欧拉方法），在相同的时间步长下，能更准确地近似求解常微分方程的解。例如，对于简单的谐振子问题（\(\ddot{x}= - \omega^{2}x\)），Leapfrog算法可以很好地保持系统的能量和周期特性，其误差随着时间步长\(\Delta t\)的平方减小。
   - **计算效率高**：它在每一步只需要计算一次力函数\(f(x)\)，对于复杂的物理系统，力函数的计算可能是计算量最大的部分。与其他高阶方法相比，Leapfrog算法在计算成本和精度之间取得了较好的平衡。
   - **保持物理量守恒特性**：在哈密顿系统中，Leapfrog算法能够很好地保持系统的能量等物理量守恒。例如，在天体力学中，用于模拟行星运动时，该算法可以使行星的总能量（动能加势能）在较长时间的模拟中保持相对稳定，这对于准确模拟物理系统的长期行为非常重要。

- **缺点**
   - **时间步长敏感性**：虽然它是二阶精度的方法，但如果时间步长\(\Delta t\)选择不当，可能会导致数值不稳定。例如，如果\(\Delta t\)过大，系统的能量可能会出现不合理的增长或衰减，模拟结果会失真。
   - **复杂边界条件处理困难**：对于一些具有复杂边界条件的问题，如粒子在有边界限制的容器内运动，并且边界条件涉及到速度或加速度的突变，Leapfrog算法可能需要额外的处理来确保准确性。

- **应用领域**
   - **天体物理学**：用于模拟行星、恒星等天体的运动。例如，在模拟太阳系中行星的轨道演化，以及星系中恒星的动力学过程等方面都有广泛应用。
   - **分子动力学**：在研究分子系统的运动和相互作用时，如模拟蛋白质分子的折叠过程，或者分子在溶液中的扩散等，Leapfrog算法可以有效地求解分子的运动方程，计算分子的位置和速度随时间的变化。

**Lie group derivative**
- **定义**
   - “Lie group derivative”通常指李群（Lie group）上的导数。李群是一种既有群结构又有光滑流形结构的数学对象，并且群运算（乘法和求逆）都是光滑映射。李群导数主要用于研究李群上的函数（或向量场）的变化率。
   - 设 $G$ 是一个李群，对于一个定义在李群 $G$ 上的函数 $f:G\rightarrow\mathbb{R}$（或更一般地，到一个向量空间），李群导数衡量了函数 $f$ 在李群元素沿着某一方向“移动”时的变化情况。

- **与普通导数的联系和区别**
   - **联系**：和普通函数在欧几里得空间上的导数类似，李群导数的目的也是描述函数的变化率。在概念上，它们都试图回答“当自变量有一个小的变化时，函数值如何变化”的问题。
   - **区别**：普通导数是基于欧几里得空间的加法结构来定义的，而李群导数是基于李群的群结构和流形结构。在欧几里得空间中，点的移动是通过简单的加法（如 $x\rightarrow x + h$）来描述的，而在李群中，元素的移动是通过群乘法（如 $g\rightarrow g\cdot h$，其中 $g,h\in G$）来实现的。此外，李群的拓扑结构和群运算的复杂性使得李群导数的计算比普通导数更具挑战性。

- **在李群表示理论中的作用**
   - 在李群表示理论中，李群导数起着关键的作用。李群表示是指李群在向量空间上的同态（保持群结构的线性映射）。当研究李群表示的微小变化或微分性质时，需要用到李群导数。
   - 例如，对于一个李群 $G$ 的表示 $\rho:G\rightarrow GL(V)$（其中 $GL(V)$ 是向量空间 $V$ 上的一般线性群），可以通过李群导数来定义李群表示的微分，这对于研究表示的分类、分解以及构造新的表示等问题都非常重要。

- **在微分几何和物理中的应用**
   - **微分几何方面**：李群是微分几何中的重要研究对象。李群导数用于定义李群上的联络（connection）和协变导数（covariant derivative）。这些概念对于研究李群的几何性质，如曲率、挠率等是必不可少的。
   - **物理方面**：在理论物理中，特别是在规范场论和量子场论中，李群有着广泛的应用。李群导数用于描述物理系统在李群对称变换下的变化。例如，在电磁学中，电磁势的规范变换形成一个阿贝尔李群（U(1) 李群），李群导数可以帮助描述电磁势在规范变换下的变化情况，这对于理解电磁相互作用的对称性和守恒定律非常重要。

**spinor field**
1. **定义**
   - 旋量场（spinor field）是一种在相对论性量子场论和数学物理中广泛使用的概念。从数学角度讲，旋量是一种特殊的对象，用于描述具有自旋（spin）特性的粒子的量子态。旋量场则是在时空每一点都定义了一个旋量的场。
   - 例如，在量子力学中，电子是自旋为1/2的粒子，其量子态可以用旋量来表示。当考虑电子在空间和时间中的分布和变化时，就需要用到旋量场，它可以描述电子在整个时空范围内的自旋状态的变化情况。

2. **与自旋的联系**
   - 自旋是基本粒子的一种内禀属性，类似于粒子的“自转”，但这种自转的概念是一种类比，实际上自旋是量子力学的概念。旋量场的主要作用是准确地描述具有自旋的粒子的状态。
   - 对于自旋为1/2的粒子（如电子、夸克等），旋量场是一个二分量的复向量场（在二维复空间中）。在相对论性量子力学中，狄拉克方程（Dirac equation）是描述自旋为1/2粒子的相对论性运动方程，旋量场就是狄拉克方程中的场变量，它的每个分量都有其物理意义，并且在洛伦兹变换（相对论中的时空坐标变换）下按照一定的规则变换，这种变换规则体现了粒子的自旋特性。

3. **在物理理论中的角色**
   - **相对论性量子场论**：旋量场是构建相对论性量子场论的基本要素之一。在这个理论框架中，所有的基本粒子都被看作是相应量子场的激发态。对于自旋为1/2的费米子（如电子、中微子等），它们的量子场就是旋量场。通过对旋量场进行量子化，可以研究粒子的产生、湮灭以及相互作用等物理过程。
   - **规范场论**：在规范场论（如量子色动力学用于描述强相互作用，电弱统一理论用于描述电磁和弱相互作用）中，旋量场与规范场（如胶子场、光子场和弱相互作用的中间玻色子场）相互作用。旋量场在规范变换下的协变性质（即按照一定规则与规范场共同变化，以保证物理规律的不变性）是研究这些相互作用的关键。例如，在量子电动力学中，电子的旋量场与光子场相互作用，这种相互作用决定了电子的电磁行为，如散射、吸收光子等。

4. **变换性质**
   - 在洛伦兹变换下，旋量场具有特殊的变换性质。对于自旋为1/2的旋量场，它按照旋量表示的洛伦兹群变换规律进行变换。这种变换不是像矢量场（如电磁场）那样简单的线性变换，而是一种更复杂的、涉及到矩阵乘法的变换。
   - 例如，在一个简单的洛伦兹变换（如沿x轴方向的速度变换）下，旋量场的两个分量会以一种特定的方式相互混合，这种混合方式是由洛伦兹变换矩阵和旋量的性质共同决定的，并且这种变换保证了物理规律在不同惯性系中的协变性。

**symplectic integration**
1. **定义**
   - 辛积分（symplectic integration）是一种用于求解哈密顿系统（Hamiltonian system）的数值积分方法。哈密顿系统是物理学中常见的动力系统，它由广义坐标（generalized coordinates）和广义动量（generalized momenta）来描述，并且系统的动力学由哈密顿函数（Hamiltonian function）决定。
   - 从数学角度看，辛积分方法能够在数值求解过程中保持系统的辛结构（symplectic structure）。辛结构是一种几何结构，与哈密顿系统中的相空间（phase space，由广义坐标和广义动量张成的空间）的体积和几何形状的保持有关。

2. **基本原理**
   - 对于一个哈密顿系统，其运动方程可以写成\(\dot{q} = \frac{\partial H}{\partial p}\)和\(\dot{p}= - \frac{\partial H}{\partial q}\)，其中\(q\)是广义坐标，\p是广义动量，\(H(q,p)\)是哈密顿函数。
   - 辛积分方法的核心是构造特定的数值算法，使得在每一步积分过程中，相空间的辛结构得以保持。例如，常用的辛算法有蛙跳算法（leap - frog algorithm），它通过交错地更新广义坐标和广义动量，在一定程度上保持了系统的辛结构。具体来说，在一个小的时间步长\(\Delta t\)下，先更新动量\(p_{n + \frac{1}{2}} = p_{n}-\frac{\Delta t}{2}\frac{\partial H}{\partial q}(q_{n})\)，然后更新坐标\(q_{n + 1}=q_{n}+\Delta t\frac{\partial H}{\partial p}(p_{n+\frac{1}{2}})\)，最后更新动量\(p_{n + 1}=p_{n+\frac{1}{2}}-\frac{\Delta t}{2}\frac{\partial H}{\partial q}(q_{n + 1})\)。

3. **重要性和优点**
   - **物理量守恒**：辛积分能够很好地保持哈密顿系统中的物理量守恒。例如，对于一个保守的哈密顿系统（能量守恒的系统），辛积分方法在长时间的数值模拟过程中能够使得系统的能量保持相对稳定，而不像一些非辛的数值方法会导致能量的漂移或者错误的积累。
   - **长期准确性**：由于保持了辛结构，在对哈密顿系统进行长期的动力学模拟时，辛积分方法可以提供更准确的结果。例如，在天体力学中，用于模拟行星的轨道演化等长期过程时，辛积分可以有效地减少误差的积累，更真实地反映系统的动力学行为。

4. **应用领域**
   - **天体物理**：用于模拟太阳系中行星、卫星等天体的运动。由于天体运动通常可以用哈密顿系统来描述，并且需要长期的精确模拟，辛积分方法可以很好地保持天体系统的能量和角动量等物理量，从而准确地模拟天体的轨道变化、潮汐锁定等现象。
   - **分子动力学**：在研究分子系统的运动和相互作用时，分子的动力学也可以看作是哈密顿系统。辛积分可以用于模拟分子的位置和动量的变化，特别是在模拟生物大分子（如蛋白质、DNA）的构象变化等长期过程中，能够更准确地追踪分子的运动状态。

**狄拉克矩阵**
1. **定义与背景**
   - 狄拉克矩阵（Dirac matrices）也称为伽马矩阵（$\gamma$ - matrices），是狄拉克方程中的重要组成部分。狄拉克方程是相对论性量子力学中描述自旋 - 1/2粒子（如电子）的基本方程。
   - 狄拉克矩阵是一组满足特定反对易关系的4×4复矩阵，在相对论性量子场论和高能物理的数学表述中起着关键作用。它们通常用$\gamma^{\mu}$表示，其中$\mu = 0,1,2,3$，分别对应于时间和三个空间方向。

2. **代数性质**
   - 狄拉克矩阵满足反对易关系：$\{\gamma^{\mu},\gamma^{\nu}\}=\gamma^{\mu}\gamma^{\nu}+\gamma^{\nu}\gamma^{\mu}=2g^{\mu\nu}I$，其中$g^{\mu\nu}$是闵可夫斯基度规（Minkowski metric），在通常的约定下，$g^{00}=1$，$g^{11}=g^{22}=g^{33}=-1$，$I$是4×4单位矩阵。
   - 这种反对易关系是狄拉克矩阵最重要的代数性质。例如，在狄拉克方程的推导和运算中，这种关系用于保证方程的相对论协变性，即保证物理规律在不同惯性系（由洛伦兹变换联系）下具有相同的形式。

3. **在狄拉克方程中的作用**
   - 狄拉克方程的形式为$(i\gamma^{\mu}\partial_{\mu}-m)\psi = 0$，其中$\psi$是旋量场（spinor field），代表自旋 - 1/2粒子的量子态，$m$是粒子的质量，$\partial_{\mu}$是对时空坐标的偏导数（$\partial_{0}=\frac{\partial}{\partial t}$，$\partial_{i}=\frac{\partial}{\partial x^{i}}$，$i = 1,2,3$）。
   - 狄拉克矩阵在这里起到了将时空导数和粒子的自旋性质相结合的作用。它们使得狄拉克方程能够同时描述粒子的相对论性运动和自旋特性。例如，在电子的相对论性量子力学中，狄拉克方程通过狄拉克矩阵正确地预测了电子的自旋磁矩等物理量。

4. **物理意义和应用**
   - **描述自旋 - 1/2粒子的行为**：狄拉克矩阵能够准确地描述自旋 - 1/2粒子在相对论性情况下的各种物理行为，包括在电磁场中的运动、散射过程等。在量子电动力学（QED）中，狄拉克矩阵用于构建电子与光子相互作用的理论框架，从而精确地计算电子的电磁性质，如电子 - 光子散射截面等。
   - **与洛伦兹变换的联系**：狄拉克矩阵在洛伦兹变换下具有特定的变换性质，这使得它们能够保证狄拉克方程的洛伦兹协变性。通过研究狄拉克矩阵在洛伦兹变换下的变换规则，可以深入了解自旋 - 1/2粒子的相对论性变换规律，进而研究在不同惯性系中粒子的物理性质。
   - **在其他物理理论中的扩展应用**：狄拉克矩阵的概念也被推广到其他物理理论中，如超对称理论（supersymmetry theory）。在超对称理论中，狄拉克矩阵的结构被用于构建超对称伙伴（superpartner）之间的关系，以实现费米子和玻色子之间的对称变换，这有助于构建更加统一的物理理论框架。

**lattice**
1. **定义与基本概念**
   - 在量子色动力学（QCD）中，“lattice”（格点）是一种用于数值计算的工具。格点QCD将时空看作是离散的晶格结构，就像一个由许多小方块（在空间维度）和小段（在时间维度）组成的网格。
   - 这种离散化的时空晶格为研究QCD提供了一种非微扰（non - perturbative）的方法。在连续时空的QCD理论中，由于强相互作用的复杂性，特别是在低能区域，很难用传统的微扰理论（基于小参数展开的理论）进行计算。而格点QCD则是把时空离散化后，在这个离散的格点结构上定义夸克场和胶子场等物理量来进行数值计算。

2. **格点上的场与粒子表示**
   - **夸克场**：在格点上，夸克场是定义在晶格节点上的场。夸克是费米子，其自由度可以通过在格点节点上的一些复数（在量子场论的表述中）来表示。这些复数的取值和变换规则反映了夸克的物理性质，如自旋、味（flavor）等属性。
   - **胶子场**：胶子场与格点的连接（link）有关。胶子是传递强相互作用的规范玻色子，在格点QCD中，胶子场可以通过定义在格点之间连接（link）上的矩阵来表示。这些矩阵的变换性质体现了胶子场的规范对称性，即保证在局部规范变换下物理规律不变的性质。

3. **作用量与格点计算原理**
   - 在格点QCD中，需要定义一个作用量（action），它类似于连续时空QCD中的作用量，用于描述系统的动力学。格点作用量通常是格点上夸克场和胶子场的函数，并且包含了格点间距（lattice spacing）等信息。
   - 通过对这个作用量进行数值处理，例如使用蒙特卡洛（Monte Carlo）方法，可以模拟夸克和胶子在格点上的行为。具体来说，蒙特卡洛方法会在格点上随机生成夸克场和胶子场的配置，然后根据作用量计算这些配置的概率，经过大量的抽样和统计，就可以计算出一些物理量，如强子的质量、强相互作用耦合常数等。

4. **优势与局限性**
   - **优势**：
     - **非微扰计算**：能够处理强相互作用的低能区域，这是连续时空QCD微扰理论难以处理的部分。例如，在研究质子和中子的内部结构（如夸克和胶子的分布）、夸克禁闭（quarks confinement）现象等方面发挥了重要作用。
     - **数值可操作性**：将复杂的QCD问题转化为可以在计算机上进行数值模拟的问题。随着计算机技术的发展，格点QCD的计算精度和可研究的物理问题范围不断扩大。
   - **局限性**：
     - **格点间距和有限体积效应**：格点间距不能无限小，有限的格点间距会导致离散化误差。同时，模拟的格点体积有限，会产生有限体积效应，这可能会影响计算结果与真实物理情况的符合程度。
     - **计算资源需求巨大**：格点QCD计算需要大量的计算资源，包括强大的计算机处理器和大量的内存。随着格点间距变小和模拟体积增大，计算量会呈指数级增长。

**we call this sublattice the bulk**
**In addition, the GPU holds a copy of the outermost borders of neighboring sublattices, which we call the halo**

# Gaugefield 结构解析

```cpp

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R18>
class Gaugefield : public SiteComm<floatT, onDevice, SU3Accessor<floatT, comp>, SU3<floatT>,EntryCount<comp>::count, 4, All, HaloDepth>
{
protected:
    SU3array<floatT, onDevice, comp> _lattice;
}

template<class floatT, bool onDevice, CompressionType comp = R18>
class SU3array : public stackedArray<onDevice, COMPLEX(floatT),EntryCount<comp>::count> {
}

template<bool onDevice, class entryType, int entryCount>
class stackedArray {

private:
    size_t _arraySize;
    gMemoryPtr<onDevice> _memory;
}

template<bool onDevice>
class gMemoryPtr {
private:
    /// It shouldn't be possible to modify the private members in here without the MemoryManagement.
    MemoryManagement::gMemory<onDevice> *raw;  /// Raw pointer to gMemory object.
    std::string name;
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R14, CompressionType compLvl1 = R18, CompressionType compLvl2 = R18, CompressionType compNaik = U3R14>
class HisqSmearing {
private:
    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge_base;
    Gaugefield<floatT, onDevice, HaloDepth, compLvl1> _gauge_lvl1;
    Gaugefield<floatT, onDevice, HaloDepth, compLvl2> &_gauge_lvl2;
    Gaugefield<floatT, onDevice, HaloDepth, compNaik> &_gauge_naik;
    Gaugefield<floatT, onDevice, HaloDepth> _dummy;
    SmearingParameters<floatT> _Lvl1 = getLevel1Params<floatT>();
    SmearingParameters<floatT> _Lvl2;

    staple<floatT, HaloDepth, comp, 3> staple3_lvl1;
}

template<class floatT, size_t HaloDepth, CompressionType comp, int linkNumber, int partNumber = 0>
class staple {
private:
    SU3Accessor<floatT, comp> _gAcc;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 1, size_t NStacks_blockdim = 1>
class HisqDSlash : public DSlash<Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>,
        Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> > {

    using SpinorRHS_t = Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks>;
    using SpinorLHS_t = Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>;

    template<CompressionType comp>
    using Gauge_t = Gaugefield<floatT, onDevice, HaloDepthGauge, comp>;

    Gauge_t<R18> &_gauge_smeared;
    Gauge_t<U3R14> &_gauge_Naik;


    //! Optimization: The memory of this spinor may be shared. However, it must not share the Halo Buffers
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> _tmpSpin;

    double _mass;
    floatT _mass2;
    floatT _c_3000;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks = 1>
class Spinorfield : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>
{
private:
    Vect3array<floatT, onDevice> _lattice;
    LatticeContainer<onDevice,COMPLEX(double)> _redBase;
    LatticeContainer<onDevice,double> _redBase_real;
}

template<class floatT, bool onDevice>
class Vect3array : public stackedArray<onDevice, COMPLEX(floatT), 3> {
}

template<bool onDevice, typename elemType>
class LatticeContainer : public RunFunctors<onDevice, LatticeContainerAccessor>  {
private:
    CommunicationBase    &comm;
    gMemoryPtr<onDevice> ContainerArray; /// Points to the array holding your data.
    gMemoryPtr<onDevice> HelperArray;
    gMemoryPtr<onDevice> ReductionResult;
    gMemoryPtr<false>    ReductionResultHost;
    gMemoryPtr<onDevice> d_out;
    gMemoryPtr<false>    StackOffsetsHostTemp;
    gMemoryPtr<onDevice> StackOffsetsTemp;
}

```

# RHMC 算法分析
```

rhmc::update
  integrator.integrate
    SWleapfrog
      updateP_fermforce
      ip_dot_f2_hisq.updateForce
        make_f0
          _cg.invert
            dslash.applyMdaggM
              applyMdaggM_nostack
                spinorOut.template iterateOverBulk<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
                  HisqDslashFunctor

class rhmc
  integrator<floatT,onDevice,All,HaloDepth,HaloDepthSpin> integrator;

class integrator
  HisqForce<floatT, onDevice, HaloDepth, 4> ip_dot_f2_hisq

class HisqForce
  AdvancedMultiShiftCG<floatT, rdeg> &_cg;

```
