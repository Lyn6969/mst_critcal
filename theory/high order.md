# Higher-order topology for collective motions

Zijie Sun1 $\cdot$ Tianjiang ${ \mathsf { H } } { \mathsf { u } } ^ { 2 } ( \mathbb { P } )$

Received: 27 June 2024 / Accepted: 2 October 2024  
$\circledcirc$ The Author(s) 2024

# Abstract

Collective motions are prevalent in various natural groups, such as ant colonies, bird flocks, fish schools and mammal herds. Physical or mathematical models have been developed to formalize and/or regularize these collective behaviors. However, these models usually follow pairwise topology and seldom maintain better responsiveness and persistence simultaneously, particularly in the face of sudden predator-like invasion. In this paper, we propose a specified higher-order topology, rather than the pairwise individual-to-individual patten, to enable optimal responsivenes-persistence rade-off i collective motion. Then, interactions in hypergraph are designed between both individuals and sub-groups. It not only enhances connectivity of the interaction network but also mitigates its localized feature. Simulation results validate the effectiveness of the proposed approach in achieving a subtle balance between responsiveness and persistence even under external disturbances.

Keywords Collective motion $\cdot$ Hypergraph $\cdot$ Higher order interaction $\cdot$ Responsiveness $\cdot$ Persistence $\cdot$ Flocking

# Introduction

The mechanism of the emergence of collective motion [1] in animal groups (such as bird flocks [2-4], fish schools [5, 6] and artificial swarms [7-10] has attracted more and more attention from cross-disciplinary researchers in recent years. Despite many efforts devoted to deciphering the underlying dynamics of collective behaviour, the exact reason why global order emerges is yet to be discovered. One of the fascinating phenomena of collective motion is that natural groups manifest a subtle balance between responsiveness to changing environmental stimuli and persistence against noisy information. Consider bird flock as a compelling example: a startle can spread swiftly across the group allowing an agile and cohesive response to a predation threat, Yet, simultaneously, the system can keep the its resilience against irrelevant or noisy information. It has been hypothesized that animal groups operate at the critical point (or at the borderline of 'chaos') [11] where these two conflicting tendencies can be achieved simultaneously.

To model such behavior, some seminal agent-based flocking models were proposed, for instance, Vicsek Model and Inertial Spin Model, (from now on, VM and ISM respectively). These models use an alignment mechanism (a.k.a. 'social forces') that relies on averaging or summing the velocities or headings of neighbors within the interaction networks. These methods are primarily based on pairwise interaction networks, which has fundamental pitfalls in capturing collective dynamics of large group in the context of collective motion. Because the coordination of large group is largely based on consensus-like algorithms, which have fragility when the group size increases [12]. That is, as group size grows up, the algebraic connectivity (the second smallest eigen value of Laplacian) of interaction network is bounded to zeros under assumptions of bounded neighborhood (all agents have a finite number of neighbors) and finite edge weight in network, limiting the convergence of consensus of consensus-like algorithms. The failure to achieve consensus significantly impedes the coordination of collective motion, resulting in a disconnection of subgroups overtime (this phenomenon is also known as fragmentation). Simulation result Fig. 3a1 and b1 provides a typical example, showing that both VM and ISM are prone to fragmentation. To address this issue and enhance group cohesion during collective motion, additional leader weight is introduced to amplify the leader's influence, giving rise to the Will Vicsek Model and the Will Inertial Spin Model (from now on, WVM and WISM, respectively) [13], which are the modified versions of VM and ISM. Admittedly, the introduction of additional leader weight leads to better cohesion of the WVM and WISM, however, the issue of fragmentation still exists, see Fig. 3c1 and d1. This heightened leader weight results in only a modest improvement in the group's responsiveness, which underpin the necessity to increase the neighbor number. Nevertheless, in pairwise network, augmenting the number of neighbors equals to increasing the edge number, which also increases the communication cost. This approach is impractical for swarms that consist of simple agents with limited communication capacity. In contrast to pairwise methods that focus solely on inter-agent interactions, hypergraph can capture interaction between agents and subgroups, increasing each agent's interaction range while keeping each agent's neighbor number unchanged. This makes hypergraphs well-suited for coordinating large groups, as shown in Fig. 3a2, d2 that HISM and HWISM groups are fully responsive to leader.

Another prominent merit of natural swarms is the ability to maintain the group's state in the presence of noise, which is referred to as persistence. In the context of collective motion, persistence manifests as most group individuals tend to move in the same direction for a long time without deviating [14]. To achieve high persistence, it is shown that the coupling strength between individuals can be fine-tuned to render the system robust to noise, however, at the cost of a decrease of sensitivity to relevant signal [15]. In [13], a willbased hierarchy, is proposed to achieve a balance between high persistence and responsiveness. However, the authors did not elucidate the fundamental root of persistence against noise. In fact, for highly ordered system, such as a magnet where all the spins are in the same direction or a highly polarized bird flock, the fluctuations caused by noise of the system are dominated by Goldstone modes [16], which have intrinsic links with the topological structure of the underlying interaction graph, as we will discuss in the following sections. Pairwise networks exhibit a localized interaction landscape, which results in a more intense diffusion under noise disturbance and consequently lower persistence. Conversely, the information aggregation mechanism in hypergraph acts as a "kernel smoother' (as depicted in Fig. 2) to effectively mitigate the heterogeneity of the interaction landscape, and leads to a higher persistence.

Therefore, one of the most pressing open problems in modelling collective behaviour is how we can achieve better balance between responsiveness and persistence. To answer this question, we propose to introduce higher-order interaction. In recent years, researchers have begun to realize that isolated nodes and cumulation of pairwise interactions cannot accurately model complex systems and demonstrate that higher-order interactions (HOIs) involving three or more nodes at a time significantly alter the emerging dynamics of systems on complex networks, including random walks [17], synchronization [18], social contagions [19], etc. These complex systems can hardly be modelled using pairwise graph representation. The framework of hypergraph is proposed to solve this problem. Hypergraph, with hyperedges that can encode interactions between an arbitrary number of system elements, has been proven capable of describing the higherorder system with different dynamics. Since most theoretical models describing collective behaviour have assumed pairwise interaction for simplicity, which may be suboptimal [20], we consider HOIs in flocking algorithms by defining an averaging (or summing) information aggregation process for flocking algorithms in hypergraph to enhance the performance. As we will observe in the following sections of this paper, flocking algorithms based on pairwise interaction networks suffer common drawbacks: pool cohesion (often leading to group fragmentation) and pool persistence in noisy environment. However, the flocking algorithms in hypergraph achieve higher responsiveness-persistence tradeoff compared with their counterparts in the graph, for that the information aggregation process in hypergraph guarantee a high connectivity of the network which is the prerequisite for reaching consensus of the group, and that the localization of Goldstone modes of the network is alleviated, leading to higher persistence in the presence of noise.

We organize our work as follows: "Methods and algorithms" elaborates on constructing a higher-order interaction network based on hypergraphs within groups and defining the information aggregation on hypergraph. The hypergraph Laplacians are deduced for the spectral analysis. Relevant performance indicators are also described. In "Results and discussions", we report simulation results to compare the performance of flocking algorithms based on graph and hypergraph under various settings. We subsequently discuss the topological roots of flocking performance. Finally, we end the paper with some concluding remarks in "Conclusions and perspectives".

# Methods and algorithms

# Preliminaries of topology

# Definition 2.1. Graph

Consider an interaction digraph $G ( V , \ E )$ with $V \quad =$ $\{ v _ { 1 } , \cdot \cdot \cdot , v _ { N } \}$ denoting the vertex set consisting of $| V | = N$ agents, and the edge set $E = \{ e _ { 1 } , \cdot \cdot \cdot , e _ { M } \}$ consists of $| E | =$ $M$ interactions between agents in the group. The neighbor set of agent $i$ is given by $\mathcal { N } _ { i } = \{ j \in V | ( i , j ) \in E , i \in V \}$ . The interaction landscape of the group can be expressed by the adjacency matrix $\mathbf { A } = \left[ a _ { i j } \right]$ , where $a _ { i j } = w _ { i j }$ if $( i , j ) \in E$ and $a _ { i j } = 0$ otherwise. $w _ { i j }$ denotes the interaction strength between agent $i$ and $j$ We can define the degree matrix $\mathbf { D }$ which is a diagonal matrix with its diagonal entries equal to the corresponding row sums of A. Then the weighted Laplacian $\mathbf { L } ^ { G } = \left[ L _ { i j } ^ { G } \right]$ of $G$ is defined as $\mathbf { L } ^ { G } = \mathbf { D } - \mathbf { A }$ . A variant of Laplacian can be defined as $\mathbf { L } ^ { R W } = \mathbf { I } - \mathbf { D } ^ { - 1 } \mathbf { A }$ which is called the random walk Laplacian of $G$

# Definition 2.2. Hypergraph

Hypergraph is a generalization of graph and can better model the higher-order interactions between groups of more than three vertices. Formally, we can define a hypergraph $H \ : = \ : \left( V , E ^ { H } \right)$ consisting of the same vertex set with $G$ and hyperedge set $E ^ { H } = \big \{ e _ { 1 } ^ { H } , \cdot \cdot \cdot , e _ { P } ^ { H } \big \}$ If $\left. e _ { \alpha } ^ { H } \right. = 2$ for all hyperedges, $H$ reduces to a graph. If $\left. e _ { \alpha } ^ { H } \right. = k$ for all hyperedges, then $H$ is a $k$ -uniform hypergraph. A hypergraph can be represented by incidence matrix $\mathbf { H } = [ h _ { i \alpha } ]$ which is defined as

$$h _ { i \alpha } = { \left\{ \begin{array} { l l } { 1 { \mathrm { ~ i f ~ } } v _ { i } \in e _ { \alpha } } \\ { 0 { \mathrm { ~ o t h e r w i s e } } } \end{array} \right. }$$

The vertex degree matrix of hypergraph $H$ is defined as $\mathbf { D } _ { V } = \mathbf { H } \mathbf { H } ^ { T } \odot \mathbf { I }$ where $\odot$ represents the Hadamard product, and I the identity matrix of proper size. The $( i , i )$ th entry of $\mathbf { D } _ { V }$ denotes the number of hyperedges containing vertex $i$ . The edge degree matrix of hypergraph $H$ is defined as $\mathbf { D } _ { E } = \mathbf { H } ^ { T } \mathbf { H } \odot \mathbf { I }$ with its $( \alpha , \alpha )$ th entry counting the number of nodes in hyperedge $e _ { \alpha }$

# Construction of hypergraph

In the literature, the most common method to construct the interaction graph is $k$ -nearest neighbor (KNN) graph which is widely used to describe interaction landscape in animal groups [21], however it has drawbacks. Firstly, KNN graph does not incorporate higher-order interaction and does not guarantee network connectivity, which leads to group fragmentation. Figure 2 gives a vivid example. It can be observed that in KNN graph, node 3 and node 5 cannot exchange information directly (Fig. 2a) but can only exchange information indirectly through the "bridge"' node 4. The edge between node 3 and node 4 is a "bottleneck"' and is prone to fragmentation [22]. While in hypergraph (Fig. 2b, c), node 3 and node 5 can exchange information directly because they are in the same hyperedge $e _ { 2 }$ . Thus, the information aggregation in hypergraph can effectively increase the effective neighbor number and increase the interaction, and enhance group cohesion.

Therefore, we propose an alternative higher-order interaction structure involving both subgroups and individuals based on uniform hypergraph. The incorporation of higher-order interaction not only guarantee the connectivity and increase responsiveness, but also decrease the localization of the zero modes of the hypergraph Laplacian, which leads to a higher persistence against noise. To construct hypergraph, we generate hyperedges based on each agent's $k$ -nearest neighbor set in Euclidean space (see Fig. 1 for more details). Suppose that each node $i$ in the graph represents an individual, and $i$ is denoted by a state vector $\mathbf { x } _ { i }$ containing various information about itself. The neighbours for each node are chosen by calculating the Euclidean distance between the state vectors of each pair of individuals, and the $k$ -nearest neighbours, together with the focal individual form a hyperedge $e _ { \alpha }$ of size $| e _ { \alpha } |$ . We repeat the above steps for each node, and ultimately obtain a hypergraph of interaction.

# Information aggregation on hypergraph

Thanks to the construction of hypergraph in previous sections, we can define a new information aggregation strategy for flocking algorithm involving higher-order interaction. Before define the information aggregation on hypergraph, we summarize the information aggregation on graph. Consider a group of $N$ individuals, each individual update its state denoted by vector $\mathbf { x } _ { i } = [ x _ { 1 } , \cdot \cdot \cdot , x _ { d } ] ^ { T } \in \mathbb { R } ^ { d }$ according to the group dynamics defined as $\dot { \mathbf { X } } ( t ) = - \mathbf { L } ^ { R W } \mathbf { X } ( t )$ where $\mathbf { X } = [ \mathbf { x } _ { 1 } , \mathbf { \mu } _ { \cdot } \cdot \cdot \cdot , \mathbf { x } _ { N } ] ^ { T } \in \mathbb { R } ^ { N \times d }$ the state of the group. Everyone in a group obeys the same rule, which is an ideal situation different from actual cases. Evidence support that there exist hierarchical organization in groups of animal [23] with the presence of both informed (often referred to as leader, informed with meaningful information such as location of resource or predators) and uninformed individuals (referred to as followers who follow the states of leaders) [24]. To model a group consisting of leader and followers, we can endow heavier weight to leaders on the interaction graph as in [13] (where the additional weight assigned to leaders is called will). When we assign will weight to each individual (for example, O for uniformed agents and 1 for informed agents), we can define the will matrix is defined as $\mathbf { W } = \mathrm { d i a g } [ w _ { 1 } , \cdot \cdot \cdot , w _ { N } ]$ . The information aggregation in group with weighted leaders can be expressed as

$$\dot { \mathbf { X } } ( t ) = - \mathbf { L } ^ { W i l l } \mathbf { X } ( t )$$

where $\mathbf { L } ^ { W i l l } = \mathbf { L } ^ { R W } + \mathbf { L } \mathbf { W }$ denotes the weighted will Laplacian.

Above summarize the information aggregation process on KNN interaction graph which will be used in various flocking algorithms. Next, we define the information aggregation on hypergraph. The information aggregation on hypergraph involves 2 steps: information flow from nodes to hyperedges and then from hyperedges to nodes.

Fig.1 Diagram of constructing KNN graph and $k$ -uniform hypergraph While for hypergraph, each individual, together with its $k$ -nearest neighbased on $k$ -nearest neighbor set $( k = 3$ for both case). For KNN graph, bours, form a hyperedge. The interaction between hyperedges and each individual chooses its $k$ -nearest neighbors to interact (denoted by agents on hypergraph in expressed by the incidence matrix $\mathbf { H }$ (coledges), the interaction landscape is described by the adjacency matrix ored entries imply the existence of interaction between agent and the A (colored entries imply the existence of interaction between 2 agents). hyperedges it belongs to)

as the sum of the informed states in $e _ { \alpha }$ , that is

(1) Information flow from nodes to hyperedges. Let hyperedge $e _ { \alpha } = \{ 1 , \ldots , m \}$ consist $m$ nodes. Here we consider 2 different rules for information aggregation. First, it is about the information exchange among uninformed agents. In such case, since every agent is equal in hierarchy, the information of each hyperedge $e _ { \alpha }$ is defined by the average state of all the uniformed agent in it

$${ \bf y } _ { \alpha } ^ { i n } = \sum _ { j \in e _ { \alpha } } { \bf x } _ { j }$$

where ${ \bf y } _ { \alpha } ^ { i n }$ denotes the informed information of hyperedge ea.

(2) Information flow from hyperedges to nodes. Since each node is affected by various hyperedges formed by different KNN subgroups, the final update of state of agent $i$ is the average of all the state of hyperedges, that is

$${ \mathbf { y } } _ { \alpha } ^ { u n } = \frac { 1 } { | e _ { \alpha } | } \sum _ { j \in e _ { \alpha } } { \mathbf { x } } _ { j }$$

where ${ \bf y } _ { \alpha } ^ { u n }$ denotes the uninformed information of hyperedge ea.

While in the case of informed agents, since the information carried by leaders is often more important, we define the information of hyperedge $e _ { \alpha }$ collected from informed agents

$$\mathbf { x } _ { i } = \frac { 1 } { | E _ { i } | } \sum _ { e \in E _ { i } } \mathbf { y } _ { e }$$

where $E _ { i } = \{ e _ { k } | i \in e _ { k } \}$ denotes the set of hyperedges incident vertex $i$

Figure 2 illustrates the difference between the information aggregation on graph and hypergraph. The information aggregation on graph is straightforward, that is, each node takes the average or sum of the information flow from neighbours. In KNN graph, a widely used method is to take the average state of neighbors which is shown in Fig. 2a. As for the uninformed information in hypergraph, the information is averaged into hyperedges, then the information in hyperedges is averaged to each node as shown in Fig. 2c which is an illustration of combination of (3) and (5) (corresponding to Laplacian $\mathbf { L } _ { u h }$ ). As for the informed information in hypergraph, since such information is often crucial for collective decision-making, the information is summed into hyperedges, then the information in hyperedges is averaged to each node as shown in Fig. 2b which is an illustration of combination of (4) and (5) (corresponding to Laplacian $\mathbf { L } _ { i h }$ . It can be observed from Fig. 2b, c that the summing strategy leads to a higher aggregated state value (the value in red in parenthesis in Fig. 2b, c) than the averaging strategy. The effect of such manipulation is twofold. On the one hand, the summing strategy can emphasize the influence of leader (informed agent), and enhance the consensus towards a reference state carried by leader. On the other hand, large state value resulting from summing strategy may lead to overregulation, leading to dramatic fluctuation group state which is harmful to group state. In this paper, in order to achieve balanced performance, we choose sum-average strategy (6), (7) for informed information, and average-average strategy (8), (9) for uninformed information.

Fig. 2 Information aggregation on graph and hypergraph. The values in circle in black represent the state of a node, while the updated state is shown in red in parentheses. a Information aggregation on KNN graph $( k = 3$ ) with each node take the average state of its neighbors to update the state of itself. b Information aggregation (sum-average strategy) on hypergraph. The values in square brackets in red represent the state of  
a hyperedge. The information is first summed from node to hyperedges (magenta arrows), then the information in hyperedges is averaged to nodes (green arrow). c Information aggregation (average-average strategy) on hypergraph. The magenta arrows and the green arrows denote average strategy

The information aggregation on graph is straightforward, that is, each node takes the average or sum of the information flow from neighbours, represented by arrows in magenta. (b) In the case of a hypergraph, firstly, nodes emit information by summation or average to hyperedge they belong to, represented by arrows in green; then for each node, the information is aggregated by taking the average of the sum of the information in the hyperedges incident to it, represented by arrows in magenta.

Based on these rules of information aggregation on hypergraph, we can define the adjacency matrix and corresponding Laplacian for both informed and uninformed cases.

# Definition 2.3. Adjacency matrix and Laplacian for informed agent on hypergraph

By exploiting the incidence matrix $\mathbf { H }$ and vertex degree matrix $\mathbf { D } _ { V }$ of hypergraph, the interaction matrix for informed agents can be expressed by

$$\mathbf { A } _ { i h } = \mathbf { D } _ { V } ^ { - 1 } \mathbf { H } \mathbf { H } ^ { T }$$

The corresponding degree matrix is defined as $\mathbf { D } _ { i h } \ =$ diag $[ \mathbf { A } _ { i h } \mathbf { 1 } ]$ where 1 is column vector of all 1. Therefore, the Laplacian for informed agents on hypergraph is

$$\mathbf { L } _ { i h } = \mathbf { D } _ { i h } - \mathbf { A } _ { i h }$$

Note that $\mathbf { L } _ { i h } \mathbf { 1 } = \mathbf { 0 }$ holds for $\mathbf { L } _ { i h }$

# Definition 2.4. Adjacency matrix and Laplacian for uninformed agent on hypergraph

Similarly, by exploiting the incidence matrix $\mathbf { H }$ vertex degree matrix $\mathbf { D } _ { V }$ and edge degree matrix $\mathbf { D } _ { E }$ of hypergraph, the interaction matrix for uninformed agents can be expressed by

$$\mathbf { A } _ { u h } = \mathbf { D } _ { V } ^ { - 1 } \mathbf { H } \mathbf { D } _ { E } ^ { - 1 } \mathbf { H } ^ { T }$$

The corresponding degree matrix is expressed as $\mathbf { D } _ { u h } =$ $\mathrm { d i a g } [ \mathbf { A } _ { u h } \mathbf { 1 } ]$ . The Laplacian for uninformed agents on hypergraph is defined as

$${ \bf L } _ { u h } = { \bf D } _ { u h } - { \bf A } _ { u h }$$

Note that ${ \mathbf { L } } _ { u h } \mathbf { 1 } = \mathbf { 0 }$ holds for $\mathbf { L } _ { u h }$

# Definition 2.5. Adjacency matrix and Laplacian in the presence of both informed and unformed agent on hypergraph

When there are both informed and uninformed agents in a system, the corresponding hypergraph Laplacian is the sum of $\mathbf { L } _ { i h }$ and $\mathbf { L } _ { u h }$ , that is

$$\mathbf { L } _ { i u h } = \mathbf { L } _ { i h } \mathbf { W } + \mathbf { L } _ { u h }$$

where $\mathbf { W } = \mathrm { d i a g } [ w _ { 1 } , \cdot \cdot \cdot , w _ { N } ]$ denotes the additional leader weight matrix. In this paper, we consider that each group has one leader with additional weight $w _ { l }$

Now we can write the information aggregation on hypergraph as below:

1. Information aggregation on hypergraph among informed agents:

$$\dot { \mathbf { X } _ { i h } } ( t ) = - \mathbf { L } _ { i h } \mathbf { X } ( t )$$

2. Information aggregation on hypergraph among uninformed agents:

$$\dot { \mathbf { X } _ { u h } } ( t ) = - \mathbf { L } _ { u h } \mathbf { X } ( t )$$

3. Information aggregation on hypergraph in the presence of both informed and uninformed agents:

$$\mathbf { X } _ { i u h } ( t ) = - \mathbf { L } _ { i u h } \mathbf { X } ( t )$$

Now we have defined the information aggregation on the KNN graphs and hypergraphs. Thanks to the Laplacians of graphs and hypergraphs, we can perform spectral analysis for different flocking dynamics on pairwise and higher-order networks, and explore the higher-order effect on flocking performance.

# Flocking algorithms based on information aggregation on hypergraph

We take into account four typical flocking algorithms, the Vicsek Model (VM) [25], the Inertial Spin Model (ISM) [26], the Will Vicsek Model (WVM) [13] and the Will Inertial Spin Model (WISM) [13]. In previous works, VM, ISM, WVM and WISM are deployed in pairwise network which we summarize in detail in Appendix B. In the following, based on the information aggregation process defined above, we generalize the flocking algorithms to hypergraph. Firstly,

we consider the ISM:

$$\begin{array} { l } { { \displaystyle { \frac { \chi } { v _ { 0 } } } { \frac { \mathrm { d } ^ { 2 } \mathbf { v } _ { i } } { \mathrm { d } t ^ { 2 } } } + \chi { \displaystyle { \frac { \mathbf { v } _ { i } } { v _ { 0 } ^ { 3 } } } \left( { \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } \right) ^ { 2 } } + \frac { \eta } { v _ { 0 } } { \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { R W } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \boldsymbol { \xi } _ { i } ^ { \perp } } \ ~ } \end{array}$$

where $\mathbf { v } _ { i }$ denotes the velocity of agent $i$ $v _ { 0 }$ the magnitude of velocity which is constant for all agents, $J$ the coupling strength, $\chi$ the inertia of spin, and $\eta$ the friction coefficient. $\pmb { \xi }$ is the Gaussian noise with mean $\mu$ and variance $\sigma ^ { 2 }$ $k _ { n }$ is noise strength. The superscript $\perp$ means that only the component perpendicular to the velocity is considered. As in [13], we introduce characteristic time $\tau$ of spin to describe the decay rate of existing spin (see Appendix B for details) for the spinbased models ISM, WISM, HISM and HWISM.

The first term on the r.h.s. is known as the social force

$$F _ { i } ^ { G } = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { R W } \mathbf { v } _ { j } \right) ^ { \perp }$$

which is based on a summation of the velocity of neighbours. Thus, the social force aggregates neighbor information on graphs which we can easily extend to hypergraphs. Based on the definition of information aggregation on hypergraph (for uninformed cases), the social force is

$$F _ { i } ^ { H } = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { u h } \mathbf { v } _ { j } \right) ^ { \perp }$$

where $L _ { i j } ^ { u h }$ is the entry of the uninformed hypergraph Laplacian $\mathbf { L } _ { u h }$ defined in (9).

The social force exerted on agent $i$ is proportional to the average velocity of the hyperedge incident to $i$ . whose velocity is the average of all the nodes contained in these hyperedges, as shown in Fig. 2. Therefore, we can define a hypergraph version of ISM as follow.

# Algorithm 2.1. Hyper Inertial Spin Model (HISM)

In hypergraph, the ISM can be rewritten into:

$$\begin{array} { l } { { \displaystyle { \frac { \chi } { v _ { 0 } } \frac { \mathrm { d } ^ { 2 } \mathbf { v } _ { i } } { \mathrm { d } t ^ { 2 } } + \chi \frac { \mathbf { v } _ { i } } { v _ { 0 } ^ { 3 } } \left( \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } \right) ^ { 2 } + \frac { \eta } { v _ { 0 } } \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } \ ~ } } \\ { { \displaystyle ~ = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { u h } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \mathbf { \hat { \xi } } _ { i } ^ { \perp } } } \end{array}$$

Going to the overdamped limit, $\eta ^ { 2 } / \chi \to \infty$ and by setting $\eta = 1$ and $J = 1$ for (17), we can deduce the hypergraph version of VM as follow.

# Algorithm 2.2. Hyper Vicsek Model (HVM)

Accordingly, the VM in hypergraph can be defined as:

$$\frac { \mathrm { d } { \mathbf { v } _ { i } } } { \mathrm { d } t } = \left( - \sum _ { j } L _ { i j } ^ { u h } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \pmb { \xi } _ { i } ^ { \perp }$$

In another case where we consider there are leaders in a group, leaders are assigned an additional weight. In this case, the total social force exerted on $i$ considering additional leader weight is $F _ { i } ^ { H w i l l } = F _ { i } ^ { u H } + F _ { i } ^ { i H }$ , with $F _ { i } ^ { u H } =$ ${ \left( - J \sum L _ { i j } ^ { u h } { \bf v } _ { j } \right) } ^ { \perp } / v _ { 0 }$ /vo and FiH $F _ { i } ^ { i H } = \left( - J \sum L _ { i j } ^ { i h } { \bf v } _ { j } \right) ^ { \perp } / v _ { 0 }$ respectively. Therefore, we can define a hypergraph version of WISM and WVM as follows.

# Algorithm 2.3. Hyper Will Inertial Spin Model (HwIsM)

The WISM on hypergraph (HWISM) can be expressed as

$$\begin{array} { l } { { \displaystyle { \frac { \chi } { v _ { 0 } } \frac { \mathrm { d } ^ { 2 } \mathbf { v } _ { i } } { \mathrm { d } t ^ { 2 } } + \chi \frac { \mathbf { v } _ { i } } { v _ { 0 } ^ { 3 } } \left( \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } \right) ^ { 2 } + \frac { \eta } { v _ { 0 } } \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } \ ~ } } \\ { { \displaystyle ~ = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { H w i l l } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \tilde { \mathbf { \xi } } _ { i } ^ { \perp } } } \end{array}$$

where $L _ { i j } ^ { H w i l l }$ is the term of hypergraph will Laplacian $\mathbf { L } _ { H w i l l } = \left[ L _ { i j } ^ { H w i l l } \right]$ defined as ${ \bf L } _ { H w i l l } = { \bf L } _ { u h } + { \bf L } _ { i h }$

# Algorithm 2.4. Hyper Will Vicsek Model (HwVM)

Accordingly, the WVM on the hypergraph, that is, the Hyper Will Vicsek Model (HwVM), can be deduced from (19) on the overdamped limit $\eta ^ { 2 } / \chi \to \infty$ and by setting $\eta = 1$ and $J = 1$

$$\frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } = \left( - \sum _ { j } L _ { i j } ^ { H w i l l } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \pmb { \xi } _ { i } ^ { \perp }$$

# Measurement of cohesion and responsiveness

Natural groups manifest high cohesion under different circumstances for some reasons. Preserving group cohesion is not only crucial to reducing predation risk, but also the integration of informative personal information of group members can lead to better collective decisions [27]. In the context of flocking, group cohesion is achieved by the consensus-finding process. However, whether a group can reach a consensus in a distributed decision-making framework is determined by the connectivity of the interaction network. Specially for first-order consensus-finding, the convergence speed to consensus state is lower-bounded by the algebraic connectivity of the topology. In a word, network connectivity is vital in the distributed consensus-finding process as it is a necessary condition for achieving consensus convergence. In the case of disconnection, the agents may not be able to communicate with each other and reach an agreement on a common state or track the leader's states. Therefore, preserving the connectivity of the communication topology is crucial for the success of consensus [12]. In the context of coordinated turn initialized by one leader, the connectivity with respect to the leader determines the propagation of turning information and thus acts as an important factor for group cohesion. To measure the network connectivity, we define the relative connectivity as

$$R C = { \frac { N _ { S } } { N } }$$

where $N _ { S }$ is the size of the strongly connected component containing the leader. The relative connectivity has significant implications on the response to a turning signal of a group. As in [13], a response indicator is defined as follows to evaluate the response of a group to an informed individual (leader)

$$r ( \phi ) = \frac { 1 } { v _ { 0 } t _ { R } } \int _ { t _ { S } } ^ { t _ { S } + t _ { R } } \mathbf { V } ( t ) \cdot \widehat { \mathbf { n } } _ { \phi } \mathrm { d } t$$

where $\mathbf { V } ( t ) = \bigl ( \sum \mathbf { v } _ { i } ( t ) \bigr ) / N$ denotes the average velocity of the flock, $\widehat { \mathbf { n } } _ { \phi }$ is the unit direction vector of the informed agent, $t _ { S }$ and $t _ { R }$ are start time and simulation time respectively. $r ( \phi ) \in [ - 1 , 1 ]$ measures the influence of the informed individual on the velocity direction of the group and the higher the higher a group's responsiveness is, and thus the faster the propagation speed of informed information. As the informed leader may travel in different directions, the overall responsiveness indicator is defined as the average of $r ( \phi )$ under different headings, that is

$$R = \frac { 1 } { \pi } \int _ { 0 } ^ { \pi } \langle r ( \phi ) \rangle \mathrm { d } \phi$$

where $\langle \cdot \rangle$ denotes the average over all the performed simulations. The responsiveness $R$ reflects the comprehensive performance of a group in response to informed information.

Fig. 3 The results of collective turning of $9 0 ^ { \circ }$ by $N = 2 0 0$ particles with HWISM in hypergraph. The trajectory of informed agent (leader) is $k = 7$ . The external reference signal is carried by the informed agent denoted by red line, while that of the uninformed agents (followers) is (yellow star) initially located at the center of the group. a1-d1 Trajecto- denoted by blue line. The arrows represent the headings. e Variation of ries of groups governed by VM, ISM, WVM and WISM in KNN graph. the group responsiveness. f Variation of the relative connectivity (21) a2-d2 Trajectories of groups governed by HVM, HISM, HWVM and of the underlying interaction graph

# Measurement of persistence

One of the main merits of natural flocks is to maintain high cohesion even in the presence of noise. The directional movement in noisy environment of animal groups can be modelled by the correlated random walks where most individuals in the group tend to move in the same direction without being deviated. This property to maintain the same group travelling direction is known as persistence. Persistence is desirable for biological groups and for artificial systems in noisy environments. To quantify the persistence of a group, we study the diffusive mortality of the group in the presence of noise. In a system where particles diffuse freely the mean-square displacement $\left. \Delta \mathbf { r } ^ { 2 } \right.$ of the particles is proportional to time, that is $\left. \Delta \mathbf { r } ^ { 2 } \right. = \overdot { \mathbf { \Gamma } } D _ { A } t$ . The diffusion coefficient $D _ { A }$ measures the average mortality within a given period of time, and the larger the higher the diffusive intensity, and thus the lower the persistence. From this point of view, we define the persistence indicator of a group as

$$P = { \frac { 1 } { \sqrt { D _ { A } } } }$$

Higher $P$ implies that the group will keep the original direction of motion even in the presence of noise exerted on each individual, which is referred to as high persistence and vice versa. For ease of comparison, the persistence $P$ is normalized into [0, 1] in each batch of data, becoming normalized persistence.

Fig.4 The snapshots of collective turn of $1 6 0 ^ { \circ }$ by $N = 1 0 0 0$ parti- The snapshots of the groups governed by VM, ISM, WVM and WISM cles. a The initial positions of $N = 1 0 0 0$ particles. The leader (yellow in KNN graph after 300 iterations. b2-e2 The snapshots of the groups star) is located at the center of the group. The colors of particles are governed by HVM, HISM, HWVM and HWISM in hypergraph after extracted from the colormap with hues reflecting the heading. b1-e1 300 iterations

# Results and discussions

# Higher-order information aggregation enhance coherence and responsiveness in flocking

In this section, we conduct simulation to investigate the improvement of cohesion and responsiveness by introducing higher-order information aggregation mechanism. The values of parameter are set as follow for all simulations unless stated otherwise: $J = 2 0 0$ $\eta = 1$ $\chi = 1$ $\tau = 0 . 2$ $v _ { 0 } = 8$ $k _ { n } = 1$ , the mean and variance of Gaussian noise are $\mu = 0$ and $\sigma ^ { 2 } = 2$ respectively, the step interval $\mathrm { d } t = 0 . 1 \mathrm { ~ s ~ }$ , the simulation length is 300 steps.

In the first simulation scenario, a swarm of $N = 2 0 0$ agents are set to perform a collective turn of $9 0 ^ { \circ }$ led by one informed agent located at the center of the group initially. Preserving network connectivity is a prerequisite for the convergence of consensus in leader-follower multi-agent systems [28].

Figure 3 demonstrates that, in the cases characterized by a disconnected interaction graph, the groups is incapable of completing collective turn. In contrast, the group governed by HISM and HWISM in hypergraph complete a collective turn successfully, thereby exhibiting a higher responsiveness to the reference signal. The fundamental reason for achieving the observed performance is attributed to the enhanced relative connectivity of the interaction graph. For flocking algorithms in hypergraph, the relative connectivity is always 1, indicating that the information emitted from the leader constantly influences the whole group, thereby facilitating the achievement of consensus. Conversely, the groups in KNN graphs fragment into subgroups, which are incapable of accurately tracking the reference signal. Consequently, these groups exhibit low responsiveness. The connectivity of the interaction network is vital in the case of big-angle collective turns with large group sizes. The result of collective turn of larger angle $( 1 6 0 ^ { \circ } )$ by larger groups $( N = 1 0 0 0 )$ is shown in Fig. 4 (see supplementary videos 1-4 for more details). It can be observed that the groups in KNN graphs fail to complete a collective turn initialized by one leader and lead to group fragmentation. However, when adopting the information aggregation in hypergraph, the collective turn of $1 6 0 ^ { \circ }$ with HISM and HWISM is achieved successfully. We also investigate the case involving even larger group with $N =$ 2500 (see supplementary videos 5-8 for more details), and the results are similar. Overall, results show that he groups governed by flocking algorithms in KNN graphs always lead to fragmentation under the above-mentioned setting, as for those in hypergraph, however, the cohesion is guaranteed and responsiveness is increased significantly. These simulation results thus validate the effectiveness of increasing responsiveness by introducing higher-order interaction.

The second simulation scenario is aimed to investigate the effect of different size of neighborhood $k$ (neighbor number for graph and hyperedge size for hypergraph) on group responsiveness. A swarm of $N = 2 0 0$ agents but with different neighborhood size $k$ are set to perform a collective turn of $9 0 ^ { \circ }$ led by one informed agent located at the center of the group initially. The result is shown in Fig. 5. It can be observed that as the ratio $k / N$ increases, the responsiveness of all groups grows. Notably, the growth in responsiveness for groups of HWVM, HISM and HWISM in hypergraph is more pronounced compared to their counterparts WVM, ISM and WIsM in graph. The underlying rationale for this disparity is the enhanced connectivity afforded by hypergraphs, which allows more followers to be connected with leader, and finally strengthens the group's responsiveness.

Fig.5 Effect of neighborhood size (neighbor number for graph and hyperedge size for hypergraph) $k$ on group responsiveness. The results are reported with mean and standard error, each point is average over 6 trials. Parameters: $N = 2 0 0$

The third simulation scenario is aimed to investigate the impact of leader weight $w _ { l }$ on group responsiveness. The result is shown in Fig. 6a. It can be observed that additional weight leads to a higher responsiveness until reaching a plateau. That is because additional weight can accelerate the convergence rate of consensus [12]. Nevertheless, the influence of the leader's weight results in a slower rise in the HWVM compared to WVM. This occurs because the aggregation of higher-order information (7) dilutes the impact of the leader's weight, allowing it to extend its influence over a broader spectrum. Consequently, the HwVM ultimately achieves a higher responsiveness compared to WVM. Regarding WISM, we observe a decrease in responsiveness, which we attribute to overregulation. In contrast, by incorporating the leader's weight, HwiSM achieves the highest responsiveness than WISM, due to its extended influence range facilitated by higher-order information aggregation (7).

In summary, the results suggest that the higher-order information aggregation (7), (9) strengthens the connectivity of the underlying interaction graph, which allows the leader to exert a broader influence. This in turn, accelerates the convergence of consensus and results in increased responsiveness. Furthermore, the additional leader weight can also boost responsiveness. However, by integrating both the enhanced connectivity and the addition leader weight, the optimal responsiveness can be achieved.

# The link between interaction graph and group persistence

The diffusive mortality of a polarized coupling system is intrinsically linked to the spectral property of the Laplacian of the underlying interaction graph. The second smallest eigenvalue $\lambda _ { 2 }$ of Laplacian is also known as algebraic connectivity of a graph which characterizing the convergence performance of consensus algorithms. By construction, the row-sums of the above-mentioned Laplacians are all zero, therefore they have a zero eigenvalue and corresponding eigenvector, which is related to Goldstone mode in the literature of statistical physics [16, 29, 30] and characterizes the effect of fluctuation caused by noise. Let $\mathbf { u } _ { 1 }$ denote the left eigenvector corresponding to the eigenvalue O of the interaction graph. The $i$ -th component of eigenvector $\mathbf { u } _ { 1 }$ is also called the eigenvalue centrality of agent i. We say $\mathbf { u } _ { 1 }$ is uniform if it is delocalized. Conversely, we say $\mathbf { u } _ { 1 }$ is localized if a few of its components have much higher values than the others. To measure the degree of localization of $\mathbf { u } _ { 1 }$ , the inverse participation ratio $( I P R )$ [31] is introduced which can be denoted as

Fig. 6 Effect of leader weight on group responsiveness and persistence. leader weight $w _ { l }$ and responsiveness. The groups are set to perform The results are reported with mean and standard error, each point is collective turn of $9 0 ^ { \circ }$ . b Variation of leader weight $w _ { l }$ and normalized averaged over six trials. Parameters: $N = 2 0 0$ $k = 7$ a Variation of persistence

Fig.7 a The diffusive performance of the average group heading, (a) at different time instants. c Violin plot of the average group heading nodel: ISM, parameter: $N = 2 0 0$ $k = 7$ , 500 steps per trial, $5 0 ~ \mathrm { t r i } .$ at the end of 500 steps. The dashed line represents the initial heading. als in total. b The probability density function (PDF) of headings in P $I P R$ of $\mathbf { u } _ { 1 }$ of the underlying interaction graph

$$I P R ( { \bf u } _ { i } ) = \frac { \sum _ { k } u _ { k } ^ { 4 } } { \left( \sum _ { k } u _ { k } ^ { 2 } \right) ^ { 2 } }$$

where $u _ { k }$ denotes the $k$ -th component of $\mathbf { u } _ { i }$ . Note that we normalize each eigenvector so that $\textstyle \sum _ { k } u _ { k } ^ { 2 } = 1$ . The $I P R$ ranges in $[ 1 / n$ , 1], where the lower limit implies the most delocalized (uniform) case with all its components being equal (i.e., $u _ { k } = 1 / \sqrt { N } )$ , and the upper limit corresponds to a one-hot case where when all the entries are O except one equals 1. As mentioned above, $D _ { A }$ controls the rate of diffusion of noise in the network, and measures how much the system deviates from the original direction in a given time interval, and in a highly polarized regime, it satisfies [16]

Fig. 8 Effect of noise level on persistence. The results are reported with mean and standard error, each point is averaged over six trials. Parameters: $N = 2 0 0$ $k = 7$

$$D _ { A } \sim \sum _ { i } \left( u _ { i } ^ { 1 } \right) ^ { 2 }$$

where $u _ { i } ^ { 1 }$ denotes the ith component of $\mathbf { u } _ { 1 }$ . It indicates that the persistence of a highly polarized system depends on the localization properties of the left eigenvector $\mathbf { u } _ { 1 }$

In the 4th simulation scenario, we aim to investigate the difference of group diffusive motility governed by different model under noise disturbance. The results are shown in Fig. 7. It can be observed that, the average group heading tends to drift gradually when subjected to noise disturbance. However, the groups in hypergraph exhibit a reduced diffusion rate which can be measured by a smaller diffusive coefficient $D _ { A }$ and higher persistence. The rationale behind this phenomenon is that, the $I P R$ of $\mathbf { u } _ { 1 }$ of the interaction graph is lower for hypergraph compared to pairwise graphs, as shown in Fig. 7d. This indicates a more delocalized zero mode for hypergraph, resulting in a slower diffusion of noise, and consequently, a higher persistence.

To further explore the impact of various noise level on group persistence. The 5th simulation scenario is conducted. To screen out the impact of leader weight on persistence, this simulation scenario exclusively examines models without additional leader weight, that is, only VM, ISM, HVM and HISM are considered. The results are depicted in Fig. 8. It can be observed that, the persistence of all models decreases as the noise strength $k _ { n }$ intensifies. However, the hypergraphbased models HVM and HISM continue to demonstrate higher persistence compared to their graph-based counterparts VM and ISM, indicating a higher level of resilience against strong noise disturbances.

Overall, the results indicate that the groups in hypergraph manifest higher persistence than that in graph. The rationale behind this phenomenon is the reduced IPR of hypergraph which is a consequence of information aggregation (7), (9). This effect persists even at the presence of more intense noise.

# Higher-order information aggregation overcomes responsiveness-persistence trade-off

It is known that VM and ISM suffer a responsivenesspersistence trade-off [13]. Further, the authors proposed WVM and WISM and showed that WVM and WISM overcome the responsiveness-persistence trade-off. Here, we examine the proposed algorithms on hypergraph (HVM, HISM, HVM and HWISM) regarding responsiveness and persistence. Results show that HISM and HWISM in hypergraph can further overcome the responsiveness-persistence trade-off and achieve a much higher persistence under high responsiveness regimes than VM, WVM, ISM and WISM in graph, see Fig. 9 for details. To further validate the effectiveness of the proposed algorithms, we also investigate another typical higher-order oscillator models HPOM and HWPOM (see Appendix C for details), and the result shows that HISM and HWISM outperform HPOM and HWPOM in responsiveness-persistence plane and show a better responsiveness-persistence trade-off. It means that a better responsiveness-persistence trade-off is obtained by introducing higher-order interaction. The reason why flocking algorithms on hypergraphs can achieve superior trade-off performance is twofold. First, the connectivity is enhanced in hypergraph, which guarantees the convergence of consensus process and leads to an enhanced group cohesion and responsiveness. Second, the localization of zero modes (Goldstone modes) of the hypergraph Laplacians is decreased compared with the graph Laplacians, which implies the slight noise exerted on individuals would spread to the whole group with a lower diffusion coefficient $D _ { A }$ , leading to a higher persistence of the entire group. By combining these two advantages, a better responsiveness-persistence trade-off is achieved using the proposed higher-order interaction topology.

# Non-uniform effect on responsiveness and persistence

So far, we have primarily focused on regular graphs and uniform hypergraphs. However, a significant portion of the hypergraph space is occupied by non-uniform hypergraphs. To delve deeper into the hypergraph landscape, we consider non-uniform hypergraph to explore the impact of (non-)uniform neighborhood size on responsiveness and persistence. In the final simulation scenario, we initially construct non-uniform graph and non-uniform hypergraph for a group of $N = 2 0 0$ . The non-uniform neighborhood size $k _ { n u }$ is determined by randomly selecting an integer from the interval [2, N/15]. Subsequently, we construct uniform graph and uniform hypergraph with identical neighborhood size $k _ { u }$ such that $k _ { u } = \langle k _ { n u } \rangle$ , where $\langle k _ { n u } \rangle$ is the average neighborhood size of the previously established non-uniform graph and non-uniform hypergraph. In this simulation scenario, we set $k _ { u } = \langle k _ { n u } \rangle = 7$ This setting allows us to compare group responsiveness and persistence in non-uniform and uniform interaction topology. The results are shown in Fig. 10.

Fig. 10 Effect of (non) uniform neighborhood size on responsiveness and persistence. The results are reported with mean and standard error. Each point is averaged over 6 trials. a Group responsiveness of uniform and non-uniform neighborhood size. b Group normalized persistence of uniform and non-uniform neighborhood size. c The IPRs of $\mathbf { u } _ { 1 }$ for different interaction network. The legend of a, b and c is shown in the  
inset of a. Parameter: $N = 2 0 0$ $w _ { l } = 1$ $\langle k \rangle = 7$ for non-uniform neighbor number for VM, WVM, ISM and WISM in graph, and non-uniform hyperedge size for HVM, HWVM, HISM and HWISM in hypergraph. $k = 7$ for uniform neighbor number for VM, WVM, ISM and WISM in graph, and uniform hyperedge size for HVM, HWVM, HISM and HWISM in hypergraph

It can be observed that, the groups in non-uniform graph and non-uniform hypergraph tends to show a slightly lower responsiveness, as depicted in Fig. 10a. This reduction is due to the instability of neighborhood size. Such instability motivates the group to avoid becoming trapped in local optima where a subgroup of agents continues to be disconnected with the major part, and thus enhancing the responsiveness. As for persistence, the persistence of VM and WVM are in the same level, very close. Because VM and WVM have very limited responsiveness, which make them lazy in face of noise disturbance.

It can be observed that the IPRs of hypergraph is lower than their graph counterparts, which make the HISM, HWISM achieve higher persistence than their graph counterparts ISM and WISM. For ISM and WISM, the uniform case has a lower IPR (imply a more delocalized interaction landscape) and thus a higher persistence (as shown in Fig. 10b, c), which is consistent with the previous analysis. For those models with additional leader weights, because the additional leader weight intensifies the heterogeneity of the interaction, leading to a higher IPR (such as WISM and HWISM), and thus resulting in a lower persistence (the persistence of WISMs and HWISMs is lower than that of ISMs and HISMs).

In summary, the non-uniformity in interaction topologies introduces neighborhood instability which is favorable for responsiveness. On the other hand, the instability of neighborhood also promotes the localized nature of the interaction graph, which in turn reduces persistence.

# Conclusions and perspectives

In this paper, we introduce higher-order interaction into flocking algorithms and achieve an optimal responsivenesspersistence trade-off. The flocking models in hypergraph successfully achieve higher responsiveness than their counterparts in graph, maintaining strong cohesion during collective maneuvers initiated by a single informed agent. Notably, we observe that the groups in pairwise graph (such as VM, ISM, WVM and WISM) frequently fails to maintain connectivity, leading to group fragmentation. The fundamental reason is that higher-order information aggregation ensures the interaction network's connectivity, which is essential for the consensus process. We also investigate the factors contributing to the high persistence observed in models in hypergraph. For a highly polarized coupling system, the system's diffusive behavior in the presence of noise is primarily influenced by the Goldstone modes that are associated with the Laplacian of the interaction graph. The process of averaging (or summing) information aggregation acts as a 'kernel smoother' of the interaction landscape, reducing the heterogeneity of interactions within groups, which significantly mitigates the localization of Goldstone modes in hypergraphs. This results in a slower diffusion of the system's heading, thus enhancing persistence. By leveraging these 2 advantages, the hypergraph-based models achieve an optimal responsiveness-persistence trade-off.

Besides, we explore the parameter space to understand how different neighborhood sizes, leader weights, and noise levels impact responsiveness and persistence in collective motion. However, there are limitations in our models. Such as, the non-symmetric interaction effect [32], the system delay effect [33], and degree correlation effect [34] still require further exploration in the presence of higher-order interaction. More types of higher-order interaction topology (like simplicial complexes, directed hypergraph, etc.) have yet to be fully investigated.

Our study takes one step forward to explore the modeling of collective behavior on higher-order networks, demonstrating the capacity of these networks to provide a deeper insight of the interaction dynamics within animal groups. This understanding establishes a foundation for the creation of bio-inspired artificial swarms based on higher-order interactions. The proposed hypergraph model can be implemented by physical robotic swarm via ad hoc network technique, where robots share similar characteristics (such as position, velocity, or visual/audio information perceived by various sensors) create a subgroup (hyperedge), so that the proposed models can be implemented. Even though our models primarily focus on the characteristics of collective motion, these models are likely to have potential of explaining other processes, such as social contagion [35], neural activity [20], biological interactions [36], etc. In future work, the proposed methods can be applied to the development of highperformance artificial swarm.

# Appendix

# A. Description of supplementary videos

1. The supplementary videos are generated with parameters: $k = 1 0$ $J = 2 0 0$ $\eta ~ = ~ 1$ $\chi \ = \ 1$ $\tau \ = \ 0 . 2$ $\mu = 0 , \sigma ^ { 2 } = 2$ . Velocity magnitude is set to $\upsilon _ { 0 } = 8$ for all agents constantly. The step interval $\mathrm { d } t = 0 . 1 \mathrm { ~ s ~ }$

2. Supplementary Video 1 (.mp4 format). Collective turns of $9 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 1 0 0 0$ particles, governed by ISM, WISM, HISM and HWISM.

3. Supplementary Video 2 ( $. { \mathrm { m p } } 4 { \mathrm { } }$ format). Collective turns of $9 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 1 0 0 0$ particles, governed by VM, WVM, HVM and HWVM.

4. Supplementary Video 3 (.mp4 format). Collective turns of $1 6 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 1 0 0 0$ particles, governed by ISM, WISM, HISM and HWISM.

5. Supplementary Video 4 (.mp4 format). Collective turns of $1 6 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 1 0 0 0$ particles, governed by VM, WVM, HVM and HWVM.

6. Supplementary Video 5 (.mp4 format). Collective turns of $9 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 2 5 0 0$ particles, governed by ISM, WISM, HISM and HWISM.

7. Supplementary Video 6 (.mp4 format). Collective turns of $9 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 2 5 0 0$ particles, governed by VM, WVM, HVM and HWVM.

8. Supplementary Video 7 (.mp4 format). Collective turns of $1 6 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 2 5 0 0$ particles, governed by ISM, WISM, HISM and HWISM.

9. Supplementary Video 8 (.mp4 format). Collective turns of $1 6 0 ^ { \circ }$ triggered by a single leader at the center of the flock of $N = 2 5 0 0$ particles, governed by VM, WVM, HVM and HWVM.

# B. Flocking algorithms in graph

Collective behaviour is shown to emerge from local interactions which can be model in graph (also known as network) [37]. Individuals in a group can be represented by vertices connected by edges which model the interactions between different agents. The weights of edge encode the interaction strength. Individuals in groups gain and emit information through their links (edges) with neighbours in the interaction graph as described above, and in turn, update their states under the influence from neighbors.

We set an interaction graph $G$ based on topological interaction rule (we refer to as KNN graph in the main text) as baseline, which is reported in the seminal research on natural flocks [21]. In this case, the neighbor set each of agent $i$ is denoted by $\mathcal { N } _ { i } = \{ j \in V | j \in \mathrm { K N N } ( i ) \}$ , where $\mathrm { K N N } ( i )$ is the $k$ -Nearest Neighbor set (or topological proximity) of agent i. Nonetheless, besides the topology, another key factor, the coupling dynamics, also influences collective behaviour, such as the Vicsek Model (VM) [25], the Inertial Spin Model (ISM) [26] and their modified version, Will Vicsek Model (WVM) and Will Inertial Spin Model (WISM) as proposed in [13]. Topology and coupling dynamics are two major factors that determine the global properties of collective behaviour. In order to investigate the link between interaction topology and collective behaviour, especially collective motion, we summarize these flocking algorithms in graph as follow.

# Algorithm A.1. Inertial Spin Model (ISM)

The inertial spin model (ISM) is proposed in [26] to describe information propagation in starling flocks which can be expressed as

$$\frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } = \frac { 1 } { \chi } \mathbf { s } _ { i } \times \mathbf { v } _ { i }$$

$${ \frac { \mathrm { d } \mathbf { s } _ { i } } { \mathrm { d } t } } = \mathbf { v } _ { i } \times \left[ { \frac { J } { v _ { 0 } } } \sum _ { j } L _ { i j } ^ { G } \mathbf { v } _ { j } - { \frac { \eta } { v _ { 0 } ^ { 2 } } } { \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } + { \frac { k _ { n } { \boldsymbol { \mathfrak { \mathfrak { E } } } } } { v _ { 0 } } } \right]$$

$${ \frac { \mathrm { d } \mathbf { r } _ { i } } { \mathrm { d } t } } = \mathbf { v } _ { i }$$

where $\mathbf { s } _ { i }$ and $\mathbf { r } _ { i }$ denotes the spin and position of agent $i$ $L _ { i j } ^ { G }$ is the term of graph Laplacian. (A2) describes an exponential decay of the spin if no additional spin is injected into agent $i$ The decay rate can be expressed as by $\exp ( \mathrm { d } t / \tau )$ as in [13], where $\tau$ is the characteristic time describing the life time of existing spin $\mathbf { s } _ { i }$ . Assuming that all agents travel at a constant speed $v _ { 0 }$ , the ISM can be rewritten into a close form as

$$\begin{array} { c } { { \displaystyle \frac { \chi } { v _ { 0 } } \frac { \mathrm { d } ^ { 2 } \mathbf { v } _ { i } } { \mathrm { d } t ^ { 2 } } + \chi \frac { \mathbf { v } _ { i } } { v _ { 0 } ^ { 3 } } \left( \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } \right) ^ { 2 } + \frac { \eta } { v _ { 0 } } \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } } \\ { { = \displaystyle \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { G } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \tilde { \mathbf { \xi } } _ { i } ^ { \perp } } } \end{array}$$

# Algorithm A.2. Vicsek Model (VM)

Going to the overdamped limit of (A4), that is, $\eta ^ { 2 } / \chi \to \infty$ and by setting $\eta = 1$ and $J = 1$ for (A4), we can deduce the VM as follow:

$$\frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } = \left( - \sum _ { j } L _ { i j } ^ { G } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \pmb { \xi } _ { i } ^ { \perp }$$

# Algorithm A.3. Will Inertial Spin Model (wIsM)

Taking into account the additional weight of leader, a variant of ISM can be deduced as follow [13]:

$$\begin{array} { l } { { \displaystyle { \frac { \chi } { v _ { 0 } } \frac { \mathrm { d } ^ { 2 } \mathbf { v } _ { i } } { \mathrm { d } t ^ { 2 } } + \chi \frac { \mathbf { v } _ { i } } { v _ { 0 } ^ { 3 } } \left( \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } \right) ^ { 2 } + \frac { \eta } { v _ { 0 } } \frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } } } } \\ { { \displaystyle { \quad = \frac { 1 } { v _ { 0 } } \left( - J \sum _ { j } L _ { i j } ^ { W i l l } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \mathbf { \hat { \mathbf { g } } } _ { i } ^ { \perp } } } } \end{array}$$

where $L _ { i j } ^ { W i l l }$ is the term of the weighted will Laplacian $\mathbf { L } ^ { W i l l }$ in (2) of the main text.

# Algorithm A.4. Will Vicsek Model (wVM)

Similarly, a variant of VM considering additional leader weight can be expressed as follow [13]

$$\frac { \mathrm { d } \mathbf { v } _ { i } } { \mathrm { d } t } = \left( - \sum _ { j } L _ { i j } ^ { W i l l } \mathbf { v } _ { j } \right) ^ { \perp } + k _ { n } \pmb { \xi } _ { i } ^ { \perp }$$

# C. Higher-order phase oscillator models

To compare the proposed higher-order models HVM, HISM, HWVM and HWISM with other types of higher-order model, the higher-order phase oscillator models are implemented in the context of collective motion. Higher-order phase oscillator models, extensively researched within the realms of network science and collective dynamics, exhibit distinct global patterns from those observed in pairwise networks [34, 38]. Consequently, we opt to utilize higher-order phase oscillator models for comparative analysis. The higher-order phase oscillator model (HPOM) in hypergraph can be express as follow:

$$\frac { \mathrm { d } \theta _ { i } } { \mathrm { d } t } = \theta _ { i } + \mathrm { s i n } \Bigg ( - \sum _ { j } L _ { i j } ^ { u h } \big ( \theta _ { j } - \theta _ { i } \big ) \Bigg )$$

where $\theta _ { i }$ denotes the heading of agent $i$ The dynamics of HPOM can be expressed as doi V; + kn5, where $\boldsymbol { \omega } _ { i }$ is the angular velocity of agent $i$

Further, by introducing leader weight in hypergraph, we can obtain another variant, the higher-order will phase oscillator model (HWPOM) which can be express as

$$\frac { \mathrm { d } \theta _ { i } } { \mathrm { d } t } = \theta _ { i } + \mathrm { s i n } \left( - \sum _ { j } L _ { i j } ^ { H w i l l } \big ( \theta _ { j } - \theta _ { i } \big ) \right)$$

Supplementary Information The online version contains supplementary material available at [https://doi.org/10.1007/s40747-024-01665-z](https://doi.org/10.1007/s40747-024-01665-z).

Funding Funding was provided by Key-Area Research and Development Program of Guangdong Province (Grant no. 2024B1111060004).

Data availability The data that support the findings of this study are available from the corresponding author upon reasonable request.

# Declarations

Conflict of interest All authors state that there is no conflict of interest.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit [http://creativecomm](http://creativecomm) ons.org/licenses/by-nc-nd/4.0/.

# References

1. Tamas V, Anna Z (2012) Collective motion. Phys Rep 517(3):71-140

2. Ling H, Mclvor GE, Westley J, van der Vaart K, Vaughan RT, Thornton A, Ouellette NT (2019) Behavioural plasticity and the transition to order in jackdaw flocks. Nat Commun 10(1):5174

3. Ling H, Mclvor GE, van der Vaart K, Vaughan RT, Thornton A, Ouellette NT (2019) Costs and benefits of social relationships in the collective motion of bird flocks. Nat Ecol Evol 3(6):943-948

4. Cavagna A, Cimarelli A, GiardinaI, Parisi G, Santagati R, Stefanini F, Viale M (2010) Scale-free correlations in starling flocks. Proc Natl Acad Sci USA 107(26):11865-11870

5. Li L, Nagy M, Graving JM, Bak-Coleman J, Xie G, Couzin ID (2020) Vortex phase matching as a strategy for schooling in robots and in fish. Nat Commun 11(1):5408

6. Sosna MMG, Twomey CR, Bak-Coleman J, Poel W, Daniels BC, Romanczuk P, Couzin ID (2019) Individual and collective encoding of risk in animal groups. Proc Natl Acad Sci USA 116(41):20556-20561

7. Xiao Y, Lei X, Zheng Z, Xiang Y, Liu YY, Peng X (2024) Perception of motion salience shapes the emergence of collective motions. Nat Commun 15(1):4779

8. Zhang S, Lei XK, Peng XG, Pan J (2024) Heterogeneous targets trapping with swarm robots by using adaptive density-based interaction. IEEE Trans Rob 40(4):2729-2748

9. Lei XK, Zhang S, Xiang YL, Duan MY (2023) Self-organized multi-target trapping of swarm robots with density-based interaction. Complex Intell Sys 9(5):5135-5155

10. Wang FK, Huang JL, Low KH, Hu TJ (2023) AGDS: adaptive goal-directed strategy for swarm drones flying through unknown environments. Complex Intell Syst 9(2):2065-2080  
    |1. Munoz MA (2017) Colloquium: Criticality and dynamical scaling in living systems. Rev Mod Phys 90:031001  
    |2. Tegling E, Bamieh B, Sandberg H (2023) Scale fragilities in localized consensus dynamics. Automatica 153:111046  
    |3. Balazs B, Vasarhelyi G, Vicsek T (2020) Adaptive leadership overcomes persistence-responsivity trade-off in flocking. J R Soc Interface 17(167):20190853  
    [4. Masuda N, Porter MA, Lambiotte R (2016) Random walks and diffusion on networks. Phys Rep 716-717:1-58

11. Romanczuk P, Daniels BC (2022) Phase transitions and criticality in the collective behavior of animals-self-organization and biological function. In: Yurij H (ed) Order, disorder and criticality. World Scientific Publishing, Singapore  
    |6. Cavagna A, Giardina I, Jelic A, Melillo S, Parisi L, Silvestri E, Viale M (2017) Nonsymmetric interactions trigger collective swings in globally ordered systems. Phys Rev Lett 118(13):138003

12. Schaub MT, Benson AR, Horn P, Lippner G, Jadbabaie A (2018) Random walks on simplicial complexes and the normalized Hodge Laplacian. SIAM Rev 62:353-391

13. Zhang Y, Latora V, Motter AE (2020) Unified treatment of synchronization patterns in generalized networks with higher-order, multilayer, and temporal interactions. Commun Phys 4:195

14. Landry NW, Restrepo JG (2020) The effect of heterogeneity on hypergraph contagion models. Chaos 30(10):103117

15. Federico B, Giulia C, Iacopo I, Vito L, Maxime L, Alice P, Young JG, Giovanni P (2020) Networks beyond pairwise interactions: structure and dynamics. Phys Rep 874:1-92

16. Ballerini M, Cabibbo N, Candelier R, Cavagna A, Cisbani E, Giardina I, Lecomte V, Orlandi A, Parisi G, Procaccini A, Viale M, Zdravkovic V (2008) Interaction ruling animal collective behavior depends on topological rather than metric distance: evidence from a field study. Proc Natl Acad Sci USA 105(4):1232-1237

17. Camperi M, Cavagna A, Giardina I, Parisi G, Silvestri E (2012) Spatially balanced topological interaction grants optimal cohesion in flocking models. Interface Focus 2:715-725

18. Nagy M, Akos Z, Biro D, Vicsek T (2010) Hierarchical group dynamics in pigeon flocks. Nature 464(7290):890-893

19. Dyer JR, Johansson A, Helbing D, Couzin ID, Krause J (2009) Leadership, consensus decision making and collective behaviour in humans. Philos Trans R Soc Lond B Biol Sci 364(1518):781-789

20. Vicsek T, Czirok A, Ben-Jacob E, Cohen II, Shochet O (1995) Novel type of phase transition in a system of self-driven particles. Phys Rev Lett 75(6):1226-1229

21. Cavagna A, Castello LD, Giardina I, Grigera TS, Jelic A, Melillo S, Mora T, Parisi L, Silvestri E, Viale M, Walczak AM (2014) Flocking and turning: a new model for self-organized collective motion. J Stat Phys 158:601-627

22. Miller N, Garnier S, Hartnett AT, Couzin ID (2013) Both information and social cohesion determine collective decisions in animal groups. Proc Natl Acad Sci USA 110(13):5263-5268

23. Ren W, Beard RW (2008) Distributed consensus in multi-vehicle cooperative control. Springer, London

24. Gouda F, Skarp K, Lagerwall S (1991) Dielectric studies of the soft mode and Goldstone mode in ferroelectric liquid crystals. Ferroelectrics 113:165-206

25. Hoinka S, Dyke P, Lingham M, Kinnunen JJ, Bruun GM, Vale CJ (2017) Goldstone mode and pair-breaking excitations in atomic Fermi superfluids. Nat Phys 13:943-946

26. Murphy NC, Wortis R, Atkinson WA (2011) Generalized inverse participation ratio as a possible measure of localization for interacting systems. Phys Rev B 83(18):184206

27. Zhou YJ, Zheng ZC, Wang T, Peng XG (2024) Swarm dynamics of delayed self-propelled particles with non-reciprocal interactions. Chaos Solitons Fract 186:115302

28. Zhou YJ, Wang TH, Lei XK, Peng XG (2024) Collective behavior of self-propelled particles with heterogeneity in both dynamics and delays Chaos. Solitons Fractals 180:114596

29. Zhang YZ, Lucas M, Battiston F (2023) Higher-order interactions shape collective dynamics differently in hypergraphs and simplicial complexes. Nat Commun 14(1):1605

30. Li JY, Wu XQ, Lui JH, Lei L (2024) Enhancing predictive accuracy in social contagion dynamics via directed hypergraph structures. Commun Phys 7(1):129

31. Mayfield MM, Stouffer DB (2017) Higher-order interactions capture unexplained complexity in diverse communities. Nat Ecol Evol 1(3):0062

32. Bode NW, Wood AJ, Franks DW (2011) Social networks and models for collective motion in animals. Behav Ecol Sociobiol 65:117-130

33. Lucas M, Cencetti G, Battiston F (2020) Multiorder Laplacian for synchronization in higher-order networks. Phys Rev Res 2(3):033410

Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
