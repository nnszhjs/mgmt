# Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach

Despite their prevalence on digital platforms, recommender systems often rely disproportionately on historically popular items (e.g., products) and providers (e.g., merchants) for future recommendations, contributing to exposure disparities. What's more, the dynamic evolution of exposure disparities—the Matthew effect—can hurt the growth of new, niche, or small providers, prevent users from exploring diverse options, and undermine the sustainability of the platform ecosystem in the long run. In this paper, we introduce a platform-level, dynamic approach to diversifying recommendations while balancing recommendation accuracy. To do so, we propose a methodology that consists of two components: (1) a dynamic graph neural network that models the Matthew effect's evolutionary trends and (2) a control module that diversifies recommendations via reducing exposure disparities across items. Offline evaluations across datasets show that the methodology outperforms existing methods in balancing diversity and accuracy. Subsequently, we validate the effectiveness of our framework with a field experiment on a large on-demand service platform; relative to the platform's current production recommender system, our methodology significantly improves the diversity in recommendations without negatively affecting the platform's key performance indicators.

**Key words:** Recommender Systems, Fairness and Diversity, Platform Design, Dynamic Graph Neural Networks, Field Experiment

## 1. Introduction

Digital platforms—online markets that connect users (e.g., customers) with providers (e.g., merchants)—invest heavily in designing recommender systems to deliver personalized recommendations (e.g., items such as news articles, products, movies, videos).[^1] In terms of impact, recommender systems enable these platforms to better identify user needs (Donnelly et al. 2024), discover sales prospects (Hu et al. 2024), and drive customer purchases (Kumar and Hosanagar 2019, Lee and Hosanagar 2021, Wang et al. 2025). Reflecting its importance, the recommender systems market is projected to reach $43.8 billion by 2031 (Wire 2023).

Recommender systems often rely on techniques such as content-based filtering, collaborative filtering, or hybrid methods to calculate similarities between items, users, or both (Adomavicius and Tuzhilin 2005, Bobadilla et al. 2013). If two users share similar tastes and preferences based on their

[^1]: According to statistics from BuiltWith, more than 480,000 websites worldwide use product recommendations. See trends.builtwith.com/analytics/product-recommendations/traffic/Entire-Internet.
viewing or purchase history, a collaborative filtering recommender system will suggest items from one user's history to the other user. Yet, these similarity-based, user-centric algorithms contribute to exposure disparities across items. That is, recommender systems rely disproportionately on historically popular items for future recommendations; niche but potentially useful items rarely receive exposure opportunities (Abdollahpouri 2020, Mansoury et al. 2020, Zhang et al. 2021b). Similarly, a collaborative filtering recommender system that relies on item similarities instead of user similarities (e.g., items that have similar ratings) will overemphasize popular items, which will have more ratings across user profiles and therefore will seem more similar to more items (Wang et al. 2018). As a result, such exposure disparities imply a less balanced representation of niche items and their providers in recommendations, exacerbating the cold-start problem (Saveski and Mantrach 2014, Zhu et al. 2021a) and resulting in a lack of diversity in recommendations.[^2]

Further, once deployed in a production environment, recommender systems function dynamically, with a self-reinforcing feedback loop involving non-stationary interactions among users, items, and the system itself. In the loop, users interact with recommender systems via their item-based activities (e.g., views, clicks, reviews, purchases); subsequently, the algorithm recommends new items to the users. Over time, the evolution of such exposure disparities leads to the Matthew effect, where "the rich get richer and the poor get poorer" (Wang et al. 2023b). Indeed, in Figure A1, Web Appendix EC.1, we assembled over 28 million user-item interactions to show that exposure disparities across popular and niche items progressively widen over time across eight platforms including MovieLens, Yelp, Amazon, and Douban.

Why should digital platforms diversify recommendations? On the one hand, a lack of diversity in item recommendations can lead to negative outcomes for both providers and users, subsequently undermining the health of the platform ecosystem.[^3] For example, in e-commerce, preferential allocation toward popular products or merchants may prevent optimal consumer-product matches due to limiting exposure to novel but useful products (Fleder and Hosanagar 2009) and drive new, niche, or small merchants to exit the platform (Mladenov et al. 2020, Singh and Joachims 2018). On the other hand, diversifying recommendations offers benefits. In Web Appendix ??, we report the results of a 12-month field experiment where an on-demand service platform exogenously

[^2]: As shown in Zhao et al. (2025), item-side (provider-side) diversity and fairness have strong connections. Specifically, if new, niche, or small providers and their items gain less (more) exposure in recommendations, the representation of those providers and their items is less (more) balanced. That is, the degree of diversity is lower (higher).

[^3]: More broadly, homogeneous recommendations could have deleterious consequences for society at large. For example, polarization is exacerbated as social media recommender systems reinforce users' opinions and beliefs by limiting exposure to diverse perspectives (Cinelli et al. 2021, Levy 2021) and/or repeatedly exposing users with similar social network characteristics to one another (Santos et al. 2021). The virality of malicious content, for example, might escalate among like-minded users due to popularity-based content recommendation (Goel et al. 2023). Diversity in recommender systems is also of significant policy relevance. Regulations such as the Filter Bubble Transparency Act require large e-commerce platforms to be transparent about their recommender systems and allow users to seek or consume information outside their "filter bubble."
boosted the rank of a randomly selected group of newly joined merchants on the recommendation page and found that increasing their exposure had a positive impact on their engagement, growth, and retention on the platform. Relatedly, Kumar et al. (2024) shows that increasing diversity in recommendations on Pinterest has positive long-term implications for underrepresented user demographic groups without hurting Pinterest's key performance indicators. Collectively, such empirical evidence substantiates the premise of diversifying recommendations for new, niche, or underrepresented groups on digital platforms.

We encounter two salient challenges in designing a diversity-aware recommender system. First, existing methods provide little guidance on how to dynamically characterize, project, and manage the Matthew effect to improve diversity in item recommendations. Specifically, the majority of existing debiasing methods do not characterize the environment's dynamic evolution to both model increased popularity and allow for niche item exposure (for surveys, see Chen et al. 2023, Wang et al. 2023b, Zangerle and Bauer 2022). Recent efforts of correcting for the Matthew effect over time still apply static debiasing at each time step (e.g., Zhu et al. 2021b), so they are not able to project the degree of Matthew effect in future periods and do not account for operational needs such as costly model retraining and recalibrating. What's more, existing methods are often designed with little to no human-in-the-loop intervention; the balance between performance and debiasing is typically determined by a weighting parameter to be optimized as part of training. However, it is important to have human control to ensure that recommender systems are effective throughout their deployment (Tsamados et al. 2024) and explicitly incorporate Matthew effect mitigation into system design (Abbasi et al. 2018).

Second, there appears to be an explicit tradeoff between item diversity (i.e., recommending items and providers in both popular and niche categories over time) and matching accuracy (i.e., recommending items and providers most relevant to a user at the moment) (Peng et al. 2024, Zhou et al. 2010). In fact, Spotify's research team has echoed such a tension: "To recommend content urgently, a good strategy is to promote relevance (and thus discourage diversity), but...to attract and retain users, ensuring that consumption is sufficiently diverse appears important" (Anderson et al. 2020). Therefore, improving item diversity requires redefining a recommender system's objective function that balances diversity with accuracy. While there has been a steady stream of research on balancing diversity and performance in recommender systems, recent work has also looked at how the operationalization of diversity impacts whether such a tradeoff is inevitable (Wick and Tristan 2019, Peng et al. 2024). In particular, by relying on easily collected, short-term data to use for performance evaluation, it is often not clear how longer-term, managerially-relevant metrics such as retention and/or future purchases are impacted.
To address these challenges, we present a novel framework, Matthew Effect Control with Dynamic GNNs (MECo-DGNN), that (a) uses a dynamic graph neural network (DGNN) to explicitly model the Matthew effect's evolution over time and (b) includes a regularization module to mitigate the effect in accordance with operational objectives. The DGNN encodes user-item interactions—intrinsic interactions such as clicks or extrinsic interactions such as review scores—in a bipartite graph structure and learns network nodes' degree information as it evolves over time.[^4] The control module balances historical exposure distributions with target exposure at the platform level based on managerial considerations of the tradeoff between accuracy and exposure to ensure sufficient item diversity. MECo-DGNN therefore captures complex network topologies and the item nodes' degree information over time, thus modeling the dynamic user-item interaction distribution. The novel regularization module approximates a predefined item click distribution embedding by constraining the observed item click distribution according to operational priorities. We measure the distribution of exposure across providers with the Gini coefficient, a well-established measure of inequality in resource allocation. By setting a target Gini coefficient based on platform objectives and managerial considerations (e.g., historical data or sensitivity analyses), the model controls this predefined item click distribution by measuring distribution diversity and adjusting the modeled distribution increasingly closer to this predefined state. In addition, the formulation of a novel weighted loss function enables us to effectively align diversity with accuracy.

We evaluate the effectiveness of MECo-DGNN in three steps. First, we apply the MECo-DGNN framework to a proprietary dataset from an on-demand service platform (the platform) and compare the performance to a robust set of benchmarking methods. Our results show that by modeling the Matthew effect's dynamic nature, MECo-DGNN outperforms thirteen benchmark models in predictive metrics (precision, recall, and net discounted cumulative gain) and diversity metrics (average percentage tail items and Gini coefficient). Second, we replicate the offline results across three public datasets to ensure robustness. Finally, we run a field experiment with the platform to provide out-of-sample validation of MECo-DGNN. We find that relative to the platform's proprietary recommender system, MECo-DGNN significantly improves recommendation diversity without negatively affecting the platform's performance metrics such as search, clicks, and purchases.

This paper has notable methodological and substantive implications. First, we propose an applicable framework to improve platform-level item diversity in recommendations. Specifically, we demonstrate that MECo-DGNN's platform-level design outperforms existing state-of-the-art methods on aligning diversity with accuracy. The capability to predetermine a target Gini coefficient offers product managers a practical decision lever that allows for operational trade-offs between

[^4]: Without loss of generality, we will refer to user-item interactions as clicks throughout the paper.
diversity and accuracy (Shulman et al. 2023). This allows for managerial input for how best to design a platform's results using recommendations (Ghose et al. 2014). Second, we validate MECo-DGNN performance improvements with a robust benchmarking suite of ablation analyses on our platform dataset as well as additional analyses on public domain data. Third, implementing a field experiment based on MECo-DGNN provides practitioners with a substantive demonstration of the effectiveness of a diversity-aware recommender system that considers multi-dimensional objectives and improves diversity without compromising on platform key performance indicators (Dzyabura and Hauser 2019, Liu 2023, Rafieian et al. 2024, Wang et al. 2025). By combining empirical benchmarking with a field experiment, we present a robust validation of our novel machine-learning model (Wang et al. 2023a) that assesses both short- and long-term utility (Besbes et al. 2024).

We organize the rest of this paper as follows. Section 2 provides an overview of related work. Section 3 describes our dynamic graph neural network framework for diversifying recommendations. In Section 4, we present our offline benchmarking evaluation using data from the partner platform, and in Section 5 we present our benchmarking results. In Section 6 we describe our field experiment. Section 7 summarizes our findings and implications for practitioners and future research.

## 2. Related Work

This work contributes to several streams of research: recommender systems, in particular work on fairness and diversity, and modeling customer dynamics. In this section, we describe the current state of the literature across these areas, drawing from work in management, information systems, and computer science.

### 2.1. Design of Recommender Systems in Information Systems and Marketing

There is a rich stream of literature investigating recommender systems in information systems. In particular, much work has studied the impact of design decisions on outcomes for customers, producers, and platforms. Diverse recommendations are introduced "with the goal of fulfilling different aspects of a consumer's need" (Song et al. 2019, p. 3739). In a classic work, Brynjolfsson et al. (2011) show that the 80/20 Pareto Principle is not as prevalent in online platforms; instead platforms instead exhibit "long tail" distributions, which provides more share to niche products. This work provided initial evidence for variety-seeking behavior in online platforms due to search, recommender systems, and scale associated with online platforms. In addition, several early studies noted that recommender system design impacts customer purchasing decisions, and can impact whether distributions of product recommendations follow the Matthew effect or exhibit more of a long tail (Fleder and Hosanagar 2009). As online platforms enhance personalization via tools such as recommender systems, consumers are exposed to more variety and subsequently consume more (Hosanagar et al. 2014).
More recently, there has been a surge of interest in how recommender systems are implemented to facilitate more diverse recommendations to consumers. Several works have modeled outcomes of recommender system usage. Using an agent-based modeling approach, Zhang et al. (2020) uncover a "performance paradox:" repeated use of a recommender system makes it less useful in the long term. They recommend a "more holistic view of recommender system performance" needed to model "longitudinal awareness" (Zhang et al. 2020, p. 99). Hagiu and Wright (2024) found that "discoverability" is important for multisided platforms; further, more established platforms can afford more discoverability.

Several empirical studies have also found insights regarding diversity in recommendations. User preferences are important to balancing diversity, as preferences for item diversity vary across users (Song et al. 2019). Increasing product variety can increase consumer surplus and welfare (Brynjolfsson et al. 2025). In a field experiment, Wan et al. (2024) observed that a item-based collaborative-filtering model shows that customers use recommendations to find higher value and lower prices. Collaborative filtering models that rely on purchase or click data will decrease diversity (Lee and Hosanagar 2019). Works focused on the design of new models of recommendation that incorporate diversity are emergent in IS. As one example, Li and Tuzhilin (2024) propose incorporating "unexpected" recommendations (Padmanabhan and Tuzhilin 1999) and show improvements in click-through rate, whether a video is viewed, and time spent on video.

There have also been studies on design decisions of recommender systems in the marketing literature. Researchers have used analytical models to examine the relative impact of utility-based, quality-based, and relevance-based content curation algorithms on the size of consumers' social networks and the quality of content generated by creators (Berman and Katona 2020). Recommendations incentivize creators to produce higher-quality content, benefiting the platform, consumers, and content creators (Qian and Jain 2024). Zou et al. (2024) account for creators' entry decisions in their theoretical framework and show that overemphasizing quality-based recommendations could reduce the supply of content, consumers' consumption diversity, and platform payoff. Monopolist firms may use low-fit recommendations to sell their products, whereas firms in a duopolistic setup may underrecommend high-fit products to ease price competition (Berman et al. 2024).

Empirical studies have also yielded important insights in this research stream. Dzyabura and Hauser (2019) proposed a recommender system that incorporates preference-weight learning in its design and diversity, novelty, and serendipity in its objective function. Zeng et al. (2024) conducted a field experiment on a video-sharing platform and showed that restricting the recommendations from popular creators decreased time spent on watching videos but increased the number of videos produced. Wang et al. (2025) developed a framework that optimizes objectives of multiple stakeholders (e.g., consumers, sellers, and couriers) in the presence of horizontal and vertical
recommendations and implemented it on Uber Eats. A study on Pinterest found that increased diversity leads to long-term benefits and can be achieved without hurting the platform's overall engagement (Kumar et al. 2024).

Critically, across the literature, the results indicate that enhanced diversity in recommendations can benefit customers, producers, or both. Producer profits and consumer surplus are key sources of revenue for platforms (Armstrong 2006, Rochet and Tirole 2003, Shi and Raghu 2020). However, a notable gap is that none of the related works take a platform approach to handling diversity. Diversity is either considered via user preferences (e.g., for diversity or novelty) or producer categories. Such methods often require detailed user and/or item information to model characteristics, preferences, and other features. In this work, we consider supply-side diversity as a platform question and present a novel recommender system design (Abbasi et al. 2024) to enhance diversity in recommendation results.

### 2.2. Rethinking the Diversity-Accuracy Tradeoff

Increasing diversity has become a core issue of study around recommender systems. In one research stream, computer scientists have investigated diversity in recommender systems from an analytical lens. The present assumption when designing recommender systems that increase diversity is that there is a diversity-accuracy tradeoff. It is related to the so-called "price of fairness" (Bertsimas et al. 2011); by optimizing a joint objective instead of only focusing on accuracy, performance will decrease. Wang and Joachims (2021) found that it is not possible to optimize for user and item diversity without tradeoffs. On the other hand, Donahue and Kleinberg (2020) derived upper bounds for the price of fairness. In particular, under certain distributional assumptions the price of fairness goes to 0; that is, maximum fairness and utility can be achieved. Relatedly, if user preferences are diverse, then fairness can be achieved "for free" (Greenwood et al. 2024).

Recent work has shown that the accuracy-diversity tradeoff may not apply for recommender systems due to the fact that not all recommendations are consumed by the user. By reformulating the problem as a question of maximizing utility instead of accuracy, Peng et al. (2024) show that increasing diversity in search results *increases* user utility. This does not need to be done by explicit exploration a la reinforcement learning, but can be achieved by simply introducing diversity into a standard explore-exploit bandit approach (Bastani et al. 2021). However, Bastani et al. (2021) found that bandit-based approaches that take an exploration-exploitation often over-explore, with a consequence of increased customer disengagement. Relatedly, Besbes et al. (2024) argue that optimizing for accuracy in recommender systems can harm long-term utility, which is often harder to measure and therefore optimize. Through theoretical and numerical analyses, they show that mixing popular and niche content in recommendations can increase long-term utility, even though they do not directly account for it in the model.
These recent theoretical results indicate that the diversity-accuracy tradeoff is not as unavoidable as previously thought. By careful consideration of the problem formulation and/or making assumptions about demand distributions or user/item characteristics, we can reasonably expect to achieve high utility and increased diversity. In the context of multi-sided platforms, where multiple providers are competing to display items to users, a platform that matches providers with customers is incentivized to display items from a variety of providers in order to maximize the likelihood that the customer will like at least one item in the list and subsequently click/purchase it (thus increasing utility). As mentioned above, improving outcomes for producers (i.e., sales and profits) and increasing consumer surplus drive revenue for platforms (Armstrong 2006, Rochet and Tirole 2003, Shi and Raghu 2020). These results indicate that diversity can improve the metrics that benefit platforms and, as such, should be considered when a platform is designing its recommender system. Given these results, there is still a gap when considering diversity in a dynamic setting when users, items, and preferences may change over time (Zhao et al. 2025).

### 2.3. Increasing Diversity in Recommender Systems

There is a robust research stream focused on debiasing the outputs of recommender systems to improve item diversity. The most prevalent approach is static debiasing, where the recommendation model is trained offline, the bias is measured, and adjustments are made to correct it. Techniques for adjusting recommendations include unbiased learning (Park and Tuzhilin 2008, Steck 2011, 2019), ranking adjustment (Abdollahpouri et al. 2017, 2019, Wei et al. 2021), causal modeling (Zhang et al. 2021b), and popularity-opportunity balancing (Zhu et al. 2021c). Such static approaches do not account for the dynamic nature of recommender systems, where users interact with items over time. While recent work has shown via simulation the evolving nature of recommender system biases (Zhu et al. 2021b), techniques to address the problem are rare. Zhu et al. (2021b) propose gradually increasing a static debiasing correction weight at each time step. This ignores the effect's evolution and dynamic intensification as consumer-item interactions increase, which can lead to a gap between the dynamic growth of the inequality and the look-back correction from the previous timestep.

The work described above, as well as surveys of the literature in top-tier computer science journals (Chen et al. 2023, Wang et al. 2023b, Wu et al. 2023, Zhao et al. 2025), show that the majority of work done to date has focused on static debiasing, where a one-time correction is made either during training or in a post hoc step. Indeed, the survey articles are consistent in surfacing challenges in long-term item-side diversity within recommender systems—the focus of our work. Recurring challenges mentioned across the surveys include the demand for dynamic solutions that handle data collected over time (Chen et al. 2023, Wu et al. 2023, Zhao et al. 2025),
maintaining predictive performance (Chen et al. 2023, Wang et al. 2023b), and increasing diversity in recommendations (Wang et al. 2023b, Wu et al. 2023, Zhao et al. 2025).

From these studies emerges a call for dynamic and diverse recommendations that maintain predictive performance while balancing short- and long-term item-side diversity. Our work leverages the network properties inherent in user-item interaction graphs. By modeling node degree and penalizing the Matthew effect directly, we can better improve performance and ensure diversity; as the degree of popular items is restricted, those clicks will go to other items, naturally enhancing diversity. In this work, we recognize the dynamic nature of the Matthew effect as an evolving phenomenon within a complex system of user-item recommendations and develop a model to mitigate the effect and facilitate ongoing performance-diversity balance. Accordingly, our proposed framework incorporates dynamic graph neural networks (DGNNs), which can model network node information and capture network evolution over time in recommender systems.

## 3. Diversifying Recommendation: Our Framework

In this section, we present MECo-DGNN, our framework for diverse recommendations at the platform level. MECo-DGNN introduces novel components to a dynamic graph neural network (DGNN) architecture. We start by reviewing DGNN preliminaries, and then motivate our design decisions, and finally introduce MECo-DGNN.

### 3.1. Preliminaries

Customer-item interactions can be represented as a bipartite graph, where the edges represent interaction behavior. When a user rates, clicks on, or purchases an item during a time period, an interaction edge exists between the two parties. Dynamic interactions between users and items can therefore be viewed as representing a complex system evolving over time in dynamic and sometimes unpredictable ways (Milgram 1967, Strogatz 2001, Boccaletti et al. 2006, Prawesh and Padmanabhan 2021, Malgonde et al. 2020). This complex system representation of recommender systems has been adopted by recent work modeling recommendation systems to improve prediction performance (Rendle et al. 2009, Li et al. 2019a, Arthur et al. 2019, Su et al. 2021). In particular, this work operationalizes the graph formulation described above and leverages graph-theoretic techniques to identify similar users and items.

In this section, we describe the key components of graph neural networks (GNNs), specifically dynamic GNNs (DGNNs), in order to position our methodological contribution (§3.3).[^5] Recent work has shown that learning distributed representations of nodes (e.g., users and items) in the context of the graph structure leads to richer representations that improve downstream predictive

[^5]: For comprehensive overviews of graph neural networks, we refer the reader to several recent survey articles on the topic (Wu et al. 2020, 2022).
performance. In particular, with regard to recommender systems, learning node representations with GNNs incorporates information from neighboring nodes into learned representations, as well as implicitly models evolving graph statistics such as node degree, which augment prediction. Within information systems, scholars have recently applied GNNs in the design of IT artifacts for areas such as social listening for public health (Kitchens et al. 2024), retail sales forecasting (Liu et al. 2025), predicting customer engagement with sponsored social media posts from brands (Ma et al. 2024), and social network connection recommendation (Yin et al. 2023). GNNs rely on learned data embeddings to represent graph nodes. DGNNs incorporate time into their learning model via a recurrent layer such as a long short-term memory (LSTM) layer.

Algorithm 1 displays the procedure for a baseline DGNN, which we step through in detail below. We can represent a dynamic bipartite graph of user-item interactions over time as a series of graph snapshots: $G = [G_1, G_2, \ldots, G_T]$, where $G_t = (N, A_t)$ is the graph at time $t$ and $1 \leq t \leq T$, with a node set $N$ and adjacency matrix $A_t$. $N$ consists of user and item nodes; because the graph is bipartite we define $R_t \in \{0,1\}^{U \times V}$ as the user-item interaction matrix at time $t$, where $U$ and $V$ denote the number of users and items, respectively. Each entry $R_t^{uv}$ is 1 if user $u$ has interacted with item $v$ at or before time $t$ and is otherwise 0. Therefore, $A_t$ can be defined as $A_t = \begin{Bmatrix} 0 & R_t \\ R_t^\top & 0 \end{Bmatrix}$.

**Algorithm 1** A baseline dynamic graph neural network for recommender systems.
1: Initialize embedding matrices $E_U$ and $E_V$, empty hidden state sequence $Z^*$
2: **while** Training **do**
3:  **for** $t$ in $T$ **do**
4:   Calculate $z_u^l$ and $z_v^l$ across layers (Equations 2,3)
5:   Calculate $z_u^*$ and $z_v^*$ (Equation 4)
6:   Concatenate $z_u^*$ and $z_v^*$ to generate $Z_t^*$ (Equation 5)
7:   Append $Z_t^*$ to $Z^*$ (Equation 6)
8:  **end for**
9:  Pass $Z^*$ through the LSTM (Equation 7)
10: **end while**
11: Generate user-item rankings $\hat{y}_{(u,v)} = D_u^{*\top} D_{v+U}^*$

GNNs model such user-item interactions with trainable user embeddings $E_U \in \mathbb{R}^{U \times d}$ and item embeddings $E_V \in \mathbb{R}^{V \times d}$, where $d$ denotes the embedding dimension (Rendle et al. 2009, Wang et al. 2019). The embedding layer of all nodes $E = \text{concat}[E_U, E_V]$ stores the learned embeddings. That is, for any user $u$, the learned embedding vector is stored in the model: $e_u \in \mathbb{R}^d$; the learned embedding vector for any item $v$ is stored in the model: $e_{v+U} \in \mathbb{R}^d$.[^6]

[^6]: Note the offset in the concatenated embedding list.
Beyond simply learning representations of users and items, GNNs model the structure of user-item interactions (e.g., user clicks). The GNN layer further refines the node embeddings with information on higher-order connectivity and local topological structure (Wu et al. 2020). This higher order information is learned through embedding aggregation and embedding propagation.

*Embedding aggregation* To capture local topology information, GNNs must collect and consolidate data from neighboring nodes. In a user-item bipartite graph, the system aggregates user information ($z_u$) and item information ($z_v$) from neighboring nodes:

$$
z_u = \sum_{v \in N_u} \frac{1}{\sqrt{|N_u|}\sqrt{|N_v|}} e_{v+U}, \quad \quad z_v = \sum_{u \in N_v} \frac{1}{\sqrt{|N_v|}\sqrt{|N_u|}} e_u, {(1)}
$$

where $e_u$ and $e_{v+U}$ are embeddings for user $u$ and item $v$, $N_u$ and $N_v$ denote the first-hop neighbors of user $u$ and item $v$, and $|N_u|$ and $|N_v|$ denote the number of nodes in $N_u$ and $N_v$. Thus, a user's item interactions are incorporated into the user embedding, and vice versa.

*Embedding propagation* To capture higher-order interactions between users and items, multiple aggregation layers are stacked to propagate embeddings. Let $h_u^l$ and $h_v^l$ denote user $u$'s and item $v$'s embedding at layer $l$. The layer first aggregates the neighbors' node embeddings at the $l$-th layer to obtain $z_u^l$ and $z_v^l$. An update function $f(\cdot)$ then takes both the aggregated neighbor embedding and node embedding vector at the $l$-th layer as input to compute embeddings at layer $l+1$. As in prior work, our update function $f(\cdot)$ takes the form of a multilayer perceptron (MLP).[^7] By propagating $l$ layers, the node obtains multiple representations, $\{h_u^1, h_u^2, \ldots, h_u^l\}$ for user nodes and $\{h_v^1, h_v^2, \ldots, h_v^l\}$ for item nodes. The final representation is obtained by an MLP $o(\cdot)$. The concatenation of all user and item representations at time step $T$ is denoted $Z_t^*$; $Z^*$ is the sequence of representations across all time steps.

$$
z_{u,t}^l = \sum_{v \in N_{u,t}} \frac{1}{\sqrt{|N_{u,t}|}\sqrt{|N_{v,t}|}} h_{v,t}^l {(2)}
$$

$$
z_{v,t}^l = \sum_{u \in N_{v,t}} \frac{1}{\sqrt{|N_{u,t}|}\sqrt{|N_{v,t}|}} h_{u,t}^l {(3)}
$$

$$
h_{u,t}^{l+1} = f(z_{u,t}^l, h_{u,t}^l) \quad \quad h_{v,t}^{l+1} = f(z_{v,t}^l, h_{v,t}^l) {(4)}
$$

$$
z_{u,t}^* = o(h_{u,t}^1, h_{u,t}^2, \ldots, h_{u,t}^l) \quad \quad z_{v,t}^* = o(h_{v,t}^1, h_{v,t}^2, \ldots, h_{v,t}^l) {(5)}
$$

$$
Z_t^* = \text{concat}[z_{1,t}^*, \ldots, z_{U,t}^*, z_{U+1,t}^*, \ldots, z_{V,t}^*] {(6)}
$$

$$
Z^* = [Z_1^*, Z_2^*, \ldots, Z_T^*] {(7)}
$$

$Z^*$ captures user and item representations at each time step (e.g., each month). To capture temporal information such as the dynamic change in edges, the derived user and item representations at each time step are passed into an LSTM layer to model a graph representation that accounts for

[^7]: In this work, we implement the MLP with 3 layers, input and output dimensions of 128 and 64, respectively, and mean aggregation as the activation function.
long-term dependencies (Hochreiter and Schmidhuber 1997, Gers et al. 2000) as follows, where $D^* \in R^{U+V}$ is the last hidden state of the LSTM.

$$
D^* = \text{LSTM}([Z_1^*, Z_2^*, \ldots, Z_T^*]) {(7)}
$$

$D^*$ consists of learned user and item embeddings which are used to make ranking predictions. Specifically, to estimate user $u$'s preference for item $v$, we calculate the inner product of user and item representations: $\hat{y}_{(u,v)} = D_u^{*\top} D_{v+U}^*$. Recommendations for user $u$ thus consist of a rank-ordering of items based on calculated $y_{(u,v)}$ scores.[^8]

### 3.2. Measuring Item Diversity in Recommender Systems

Having established a theoretical connection between item-side exposure diversity and user utility, we now consider how item-side diversity can be identified, measured, and accounted for during model training. Recommender systems model user-item interactions, resulting in a dynamically evolving process. The interactions form a feedback loop in which the recommender system shapes user behavior, which then generates additional data for model training. This loop can cause over-recommendation of popular items, restricting recommendation opportunities for niche items that users consider valuable and satisfying for their novelty and surprise (Abdollahpouri et al. 2017). This trend increases over time; popular items continue to grow in popularity at the expense of niche items—exacerbating homogeneous recommendations due to the Matthew effect (Zhu et al. 2021c, Zhang et al. 2021b).

The Matthew effect describes a type of exposure disparity, namely a disparity in resource allocation. Researchers across fields have studied the Matthew effect, in particular investigating metrics such as the Lorenz curve to quantify its impact (e.g., measuring system inequality) (Fleder and Hosanagar 2009, Rigney 2010). Consider a population segment such as books on Amazon, rank-ordered by their number of user interactions (e.g., clicks) on an x-axis. The Lorenz curve plots the attributable percentage of a shared resource (here, clicks) on the y-axis (Park et al. 2020, Oestreicher-Singer and Sundararajan 2012, Molaie and Lee 2022). When resources are equally distributed, the curve is a line with slope 1 (i.e., $y = x$). As the allocation distribution shifts, resources move to relatively few individuals at the top of the distribution, and the curve is pulled toward the graph's bottom right corner (Figure 2, presented in Section 4). In our example, the proportion of clicks skews in favor of those books that already have a high number of clicks. A Gini coefficient of 0 indicates a Lorenz curve at the line of equality and an equal distribution of resources (e.g., no shaded region in Figure 2 below); in our case, a lower value represents more fair recommendations.[^9]

[^8]: We discuss parameter optimization for these components in Section 3.4.

[^9]: Because of heterogeneity in user preferences and item qualities, we do not expect a Gini coefficient close to 0.
Although the Gini coefficient and Lorenz curve are typically used for economic indicators such as income, researchers have applied them in prior work to describe recommender systems in terms of how they allocate resources (i.e., item recommendations) among users (Chen et al. 2023, Wang et al. 2023b). However, to date no study has offered a comprehensive analysis of the Matthew effect as a *dynamic phenomenon* in recommender systems. In this work, we use the Lorenz curve and Gini coefficient to measure item-side diversity in recommender systems; we also incorporate the Lorenz curve into our model optimization to ensure that we can correct learned representations that exacerbate the Matthew effect.

### 3.3. MECo-DGNN

Having introduced the general DGNN framework and the Lorenz curve, we now present our methodological innovations. The standard DGNN approach suffers from two issues that we address in our proposed framework. First, while DGNNs model graphs as they evolve over time, they do not model higher-level node properties, such as their degree. This trade-off allows for more generalizable latent representations at the expense of losing structural information of the focal graph network. They therefore cannot extract complex information from the evolving graphs beyond the adjacency matrices at each time step. Second, current DGNN modeling approaches do not account for or address population degree distribution shifts such as those perpetuated under the Matthew effect.

To address the first issue of modeling higher-level information, we propose a dynamic degree-regularization module (§3.3.1) that models how node features change over time. To address the second issue, we propose a novel regularization procedure penalizing learned models that deviate from a predetermined target Gini coefficient (§3.3.2). We illustrate our MECo-DGNN framework in Figure 1. With MECo-DGNN, we introduce two novel design modules: the Stage Degree Regularization module and the Matthew effect Regularization module. The modules are based on a) modeling the evolution of the Lorenz curve (and Gini coefficient) over time and b) incorporating an operational constraint on the model in the form of a predefined target Gini coefficient.

The Stage Degree Regularization module uses the Lorenz distribution at each stage to predict each item node's degree (§3.3.1) so that the model can learn the system's topological structure over time. The Matthew effect Regularization Layer penalizes the model for deviating from the predetermined Gini coefficient, thus balancing performance and diversity according to organizational priorities (§3.3.2).

#### 3.3.1. The Stage Degree Regularization Layer

GNNs are capable of learning local and global graph information through the graph adjacency matrix. However, standard GNNs can struggle to preserve structural information when learning representations (Xu et al. 2019). Prior work has included a degree-regularization constraint to both capture higher-order structural features
and ensure that the embedding process retains key information (Tu et al. 2018, Li et al. 2019b, Zhang et al. 2021a). While prior work has defined a degree regularization constraint for static GNNs (Tu et al. 2018, Zhang et al. 2021a), we extend this process to the case of DGNNs to ensure that the structural information is retained over timesteps. What's more, degree regularization has not previously been considered as a diversity-enhancement technique.

To capture the Lorenz distribution's evolution via DGNNs, we embed degree information for item nodes (i.e., books, movies, or products) over time. We define a degree-regularity constraint, $L_{reg}$, to learn node-embedding degree information as follows. Recall $Z^*$ is the sequence of learned graph representations across timesteps, and $Z_t^*$ is the representation at time $t$. We extract the latent representation for the nodes and pass them through a feed-forward neural network to predict the node degrees ($\hat{k}_v^t$).

$$
\hat{k}_v^t = \text{FFNN}(Z_{t\ v+U}^*), {(8)}
$$

$$
L_{reg}^t = \sum_{v=1}^{N} (log(d_v^t + 1) - \hat{k}_v^t)^2, {(9)}
$$

$$
L_{reg} = \sum_{t=1}^{T} L_{reg}^t. {(10)}
$$

Here, $d_v^t$ represents the number of interactions (i.e., node degree) of item $v$ at time $t$ in graph snapshot $G_t$, FFNN($x$) denotes a one-layer multi-layer perceptron feed-forward neural network estimating the (log) degree from the embedded representation $Z_{t\ v+U}^*$, and $T$ represents the total

**Figure 1** The MECo-DGNN Framework (Novel Components in Gray)

[Figure showing the MECo-DGNN architecture with T1, T2, ..., Tn time steps, each going through Embedding, GNN, and producing item degree distributions, with LSTM connecting the time steps, and Matthew regularization module leading to Item rankings. Legend shows Item embedding (triangle) and User embedding (diamond).]
number of time steps. Consistent with prior work, we add 1 to the degree number to avoid the degenerate edge case where all nodes have no edges (i.e., degree 0, Tu et al. 2018).

The intuition behind this series of equations is that by explicitly predicting node degree from the learned representations and ultimately incorporating degree prediction into the loss function, the model will be trained to generate representations that encode first-order (adjacency) and higher-order (degree) information. This is key for modeling the Matthew effect because in the bipartite graph structure, edges represent user-item interactions (i.e., clicks).

#### 3.3.2. The Matthew Effect Regularization Layer

Because the Matthew effect is dynamic, items become increasingly unevenly distributed over time, and exposure opportunities for long-tail items decrease (§3.2). Therefore, we propose a Matthew effect regularization module that constrains learning via a predetermined target Gini coefficient and the learned item-node degree distribution information. This module ensures that learned embedding representations conform to the target Gini coefficient and the interaction distribution driving the recommended results balances exposure, popularity, and operational requirements.

As discussed earlier, the Lorenz curve is a graphical representation of the cumulative distribution of a resource (here, item popularity as measured by clicks) across a population. In the context of items, the curve shows the distribution of item clicks on a customer-item bipartite graph. The least popular items are represented at the bottom left of the curve; the most popular are represented at the top right. Any Lorenz curve can be quantified by a corresponding Gini coefficient.

Specifically, the literature generally defines the Lorenz curve as $\eta = L(p_x)$, where $p_x$ is the proportion of entities receiving a resource up to amount $x$ and $\eta$ is the proportion of total resources they receive (Paul and Shankar 2020). A Lorenz curve is non-negative, defined on the range $[0, 1]$, and monotonically increasing. While multiple formulations appear in the literature, we follow a single parameter formulation: $L(p_x, \kappa) = \frac{\exp(\kappa p_x) - 1}{\exp(\kappa) - 1}$, where $\kappa > 0$ is estimated from available data, such as household income or item clicks (Chotikapanich 1993, Paul and Shankar 2020).

From the Lorenz curve, we calculate the Gini coefficient, a traditional measure of income inequality within a population. The Gini coefficient represents the proportion of the area on a graph under the line of equality (i.e., $y = x$) but not under the Lorenz curve. Given the Lorenz curve formulation, the Gini coefficient is defined as: $Gini = \frac{(\kappa - 2)e^\kappa + (\kappa + 2)}{\kappa(e^\kappa - 1)}$. The coefficient ranges between 0 and 1 and reflects the degree of inequality.

MECo-DGNN constrains the DGNN's evolution over time so that the learned item degree distribution as captured in our embeddings conforms to the predetermined Gini coefficient threshold. To control for the Matthew effect, we first make an operational decision independent of the model training process. We calculate the Gini coefficient's historical evolution in the training data to
project the desired range that the recommender system should achieve, $Gini_{target}$. Firms using recommender systems can make the same decision based on their internal goals and other non-technical considerations (Rakova et al. 2021). Given $Gini_{target}$, we analytically calculate the Lorenz curve parameter $\kappa$ using the method of least squares for the resolution of the system of nonlinear equations.[^10] Having obtained $\kappa$, we then rank the items by their popularity and plot the Lorenz curve by plugging in values $0 \leq p_x \leq 1$ (§3.2).

We now have a plot of clicks based on our target Gini coefficient that we assign to our rank-ordered items. We then divide the items into $Q$ groups according to the principle of equal proportions (Biega et al. 2018) using this adjusted node degree as our popularity measure: $S_0 = [1/Q, 1/Q, 1/Q..., 1/Q]$. For example, if $Q = 2$, we split our data at the point where the cumulative distribution of clicks as estimated from $Gini_{target}$ equals 0.5. Thus, $S_0$ is a distribution over a target item popularity distribution where each bucket holds the same probability mass. In the extreme, if $Gini_{target} = 0$, then bin sizes would be equal and resources (clicks) would be equally distributed. By defining $Gini_{target}$ as a managerial lever, firms can modulate between equal distribution and a distribution based purely on historical data according to their objectives.

$S_0$ represents a balance between popularity and exposure. To limit the Matthew effect worsening over time during training, we constrain the degree distribution for each group of products to be near $S_0$. Specifically, we reconstruct our user-item click graph from our learned embeddings. Let $C_{(u,v)}$ be the likelihood of an edge existing between user $u$ and item $v$. We use a feed forward neural network (FFNN) to estimate the probability of a connection between user $u$ and item $v$, $p(C_{(u,v)})$, and sum over all users $u \in U$ to estimate item degree.[^11] To calculate the proportion of clicks for each group, we sum over all item degrees and divide by the total number of edges in the network to obtain the distribution $S = [S_1, \ldots, S_q, \ldots, S_Q]$. We then calculate the Kullback-Leibler divergence from $S$ to $S_0$ to measure how far the true distribution diverges from the target distribution as specified by $Gini_{target}$ and $S_0$:

$$
C_{(u,v)} = \text{FFNN}(D_u^* \cdot D_{v+U}^*) {(11)}
$$

$$
d_v^* = \sum_{u \in U} p(C_{u,v}) {(12)}
$$

$$
S_q = \frac{\sum_{v \in q} d_v^*}{\sum_{v \in V} d_v^*} {(13)}
$$

$$
S = [S_1, \ldots, S_q, \ldots, S_Q] {(14)}
$$

$$
L_{mat} = \text{KLD}(S || S_0). {(15)}
$$

[^10]: We use the `fsolve` function to solve for $\kappa$.

[^11]: We instantiate the network with two layers, input and output dimension of 64 and 2, respectively, and Gumbell softmax as the activation function. We select Gumbell softmax to facilitate learning the probability of a click as opposed to a binary click/no-click labeling.
Note that the binning described above provided two benefits. First, the distributions we are comparing via KL Divergence have $Q$ elements rather than $N$ elements, which reduces computation but also bolsters the probability mass in each bin. Second, KL Divergence is sensitive to elements with probability 0 in a sample distribution; binning helps avoid such scenarios.[^12]

Our two novel modules achieve complementary aims. $L_{reg}$ brings our model in line with the observed node degree and corresponding Lorenz curve. $L_{mat}$ balances our organizational priorities (via $Gini_{target}$) with the historical data.

### 3.4. Model Optimization

To learn the MECo-DGNN model parameters, we define a loss function consisting of three parts. The first term is Bayesian Personalized Ranking loss (BPR, Rendle et al. 2009), $L_{BPR}$, which has been widely applied in the literature (Gao et al. 2023). BPR considers the relative order between observed and unobserved user-item interactions. Specifically, it assumes that observed user interactions carry more weight in reflecting user preferences than unobserved interactions:

$$
L_{BPR} = \sum_{(u,v,w) \in O} -\ln \sigma(y_{(u,v)} - y_{(u,w)}). {(16)}
$$

Here, $y_{u,v}$ are positive examples (i.e., clicks), $y_{u,w}$ are negative examples, $O = \{(u,v,w) \mid (u,v) \in R^+, (u,w) \in R^-\}$ represents training data, $R^+$ denotes interactions in the training data (positive examples), $R^-$ denotes interactions not in the training data (negative examples), and $\sigma(\cdot)$ is the sigmoid function. Thus, BPR loss seeks to maximize the difference between the predicted scores of positive examples ($y_{u,v}$) and negative examples ($y_{u,w}$). The second and third terms in our loss function are the loss functions from our novel modules (Equations 10 and 15). Combining the three leads to our overall MECo-DGNN loss function:

$$
L_{\text{MECo-DGNN}} = L_{BPR} + \alpha L_{reg} + \beta L_{mat}. {(17)}
$$

$L_{reg}$ represents the model's ability to learn degree information and the Lorenz distribution for each time step and is controlled by the parameter $\alpha$. $L_{mat}$ penalizes representations where click distributions deviate from the predefined Lorenz distribution and is controlled by parameter $\beta$.

With our proposed MECo-DGNN framework, we address two key gaps in the recommender systems literature. First, we estimate node degree over time and calculate loss with respect to the true degree to model the dynamic evolution of graph nodes' higher-level information. Second, we regularize based on a predefined Lorenz curve and corresponding Gini coefficient, which balances diversity and performance, to constrain model training, account for the principle of equal proportions, and encode operational requirements.

[^12]: Because $S_0$ is a uniform distribution, Equation 15 can be rewritten as $L_{mat} = \log Q - H(S)$ where $H(\cdot)$ is Shannon entropy.
## 4. Empirical Evaluation of MECo-DGNN

Our data come from a leading on-demand service platform in China (the platform), which offers a wide array of services such as food delivery, ride-hailing, travel bookings, and beauty and personal care. The platform provides its services through an integrated mobile app that connects users (customers) with merchants (e.g., restaurants, hotels, hair salons) in more than 3,000 regions nationwide. As of December 2022, the platform hosted over 670 million transacting users and 9 million active merchants, resulting in more than 17 billion on-demand delivery transactions.

To evaluate MECo-DGNN's potential to improve recommendation accuracy, diversity, and key performance indicators, we worked with the platform's search team, which is responsible for designing recommender systems to deliver the most relevant products and services to users when they perform a search. We collected a random sample of transaction data in 2022. For our main analyses, we use a subsample of 377,716 user-item interactions (i.e., click behaviors) across 50,000 users and 139,131 items over 5 weeks, covering a wide array of services such as food delivery, ride-hailing, travel bookings, and beauty and personal care. The average number of clicks per user is 1.99. In Figure 2, we plot the Lorenz curve and Gini coefficient over time and found the presence of evolving exposure disparities on the partner's platform, which are consistent with the data patterns on other platforms (Figure A1, Web Appendix EC.1). Next, we describe our offline evaluation of MECo-DGNN relative to a robust set of benchmark models using this proprietary data.

**Figure 2** Matthew Effect on the Partner Platform. The Gini coefficient is calculated as the ratio between the shaded area and the total area under the diagonal line (shaded area is for week 1). Week over week, the shaded area and Gini coefficient increase, exacerbating the lack of diversity.

[Figure showing Lorenz curves for 5 weeks with Gini coefficients: Week 1: 0.716, Week 2: 0.762, Week 3: 0.779, Week 4: 0.791, Week 5: 0.805. The graph shows Cumulative Percentage of Clicks vs Cumulative Percentage of Items, with curves for w1-w5 showing increasing inequality over time.]
### 4.1. Comparison Models

We conduct our benchmark analyses against thirteen comparison models to evaluate the significance of MECo-DGNN's performance improvements. Our benchmarking models cover traditional recommender systems models, recent high-performing models, and models designed for diversity. In addition, we evaluated ablations of our MECo-DGNN framework with key components removed.

To demonstrate the validity of GNNs generally, our first benchmark is a matrix factorization model, a classic model in recommender systems which has been previously applied in business contexts (Farias and Li 2019, Dhillon and Aral 2021). In addition, we include a neural collaborative filtering model (NeuralCF, He et al. 2017), a generic a dynamic graph convolutional neural network model (DyGCN, Manessi et al. 2020), and an actor-critic reinforcement learning model (Lillicrap et al. 2016, Liu et al. 2018) as baseline benchmark models.

Next, we consider three recent models from the recommender systems literature. First, Bayesian personalized ranking using matrix factorization from implicit feedback (BPRMF) is a matrix factorization model trained to minimize BPR loss (Rendle et al. 2009). Self-attentive sequential recommendation (SASRec) is a variant of Transformer using a set of trainable position embeddings to encode items in rank-order for sequential recommendation (Kang and McAuley 2018). Lastly, LightGCN is a graph convolutional neural network (GCN) model that learns user and item embeddings by linearly aggregating their neighbors on a user-item interaction graph (He et al. 2020). Notably, SASRec and LightGCN are still widely used in benchmarking despite being several years old (e.g., Zhang et al. 2024, Ren et al. 2024, Lin et al. 2024). What's more, LightGCN was until recently the platform's current status quo model in production.

To compare against static bias correction methods, we implement two postprocessing methods to BPRMF, SASRec, and LightGCN. Popularity Compensation (PC) is a state-of-the-art ranking adjustment method relying on BPR matrix factorization to control popularity bias (Zhu et al. 2021c). Popularity-bias Deconfounding and Adjusting (PDA) is a leading method of learning causal embeddings and effectively eliminates detrimental biases (Zhang et al. 2021b).[^13]

Lastly, to analyze the impact of each component in MECo-DGNN, we conduct ablation analyses where we remove components from the model. We test a static version of our model (MECo-GNN), a version without the degree regularization ($\alpha = 0$), a version without the Matthew effect regularization ($\beta = 0$), and a version where we target perfect equality with our target Gini coefficient ($Gini_{target} = 0$).

[^13]: In the SASRec-PDA implementation, we apply causal learning only to the prevalence of the most recently interacted item in each sequence.
### 4.2. Experiment Details

For time-series data, we use the user-item interaction bipartite graph from four periods (i.e., weeks 1 to 4) as inputs to predict user behavior and item distribution in week 5. We employ a random selection process for each dataset to allocate 70% of the historical user interactions as our training set, 20% as the test set, and 10% as a validation set for fine-tuning model hyperparameters. That is, 70% of the weeks 1 through 5 data is used for training, 10% is used for validation, and 20% is used for testing. In each fold, our inputs are the graph snapshots from weeks 1 through 4, and our rankings are based on new edges (i.e., clicks) in the week 5 data. We consider each observed user-item interaction (click) a positive instance. To create negative instances, we employ a negative sampling strategy by pairing the positive instance with randomly selected items that users have not previously rated or consumed. For MECo-DGNN, we set $\alpha$ to 1e-5 and $\beta$ to 100.[^14]

**Table 1** Evaluation Metrics

| Metric | Equation | Description |
|--------|----------|-------------|
| **Performance Metrics** | | |
| CG@k | $\sum_i^k G_i$ | Total gains accumulated from the top K items in the recommendation list. |
| DCG@k | $\sum_i^k \frac{G_i}{log_2(i+1)}$ | Assigns weights to relevance scores according to their positions in the ranking. |
| IDCG@k | $\sum_{i=1}^{K^{ideal}} \frac{1}{U} G_i^{ideal} \log_2(i+1)$ | Ideal DCG score, achieved by recommending the most relevant items at the top. |
| NDCG@k | $\frac{\text{DCG@k}}{\text{IDCG@k}}$ | DCG with a normalization factor in IDCG, the denominator. |
| Recall | $\frac{a}{a+c}$ | At k, measures the fraction of relevant items found within the top-k recommendations. |
| Precision | $\frac{a}{a+b}$ | At k, quantifies the ratio of recommended items in the top-k relevant recommendations. |
| **Diversity Metrics** | | |
| APT | $\frac{1}{\|U_t\|} \sum_{u \in U_t} \frac{\|i, i \in (L(u) \cap \sim \Phi)\|}{L(u)}$ | Average percentage of tail items. |
| Gini | $\frac{\sum_{v_x, v_y \in V} \|f(v_x) - f(v_y)\|}{2\|V\| \sum_v f(v)}$ | Degree of item unfairness in recommender system. |

### 4.3. Evaluation Metrics

Researchers have proposed many metrics to evaluate the performance and diversity of recommender systems (Zangerle and Bauer 2022). To evaluate prediction performance, we use normalized discounted cumulative gain (NDCG), recall, and precision. To assess diversity, we use average percentage of tail items (APT) and the Gini coefficient. Abdollahpouri et al. (2017) proposed APT as a metric to measure long-tail item coverage. Molaie and Lee (2022) propose that as the Gini coefficient decreases, the corresponding item distribution becomes increasingly balanced and long-tail items receive increasing exposure. Table 1 provides a summary of metrics.[^15] In our experiments,

[^14]: We conducted a thorough analysis on public data to validate these values in Web Appendix ??.

[^15]: For a detailed discussion of the metrics, please refer to Web Appendix EC.3.
we set the length $K$ of the recommendation list to 50 to calculate performance metrics (i.e., Precision@50, Recall@50, and NDCG@50). We ran each model 3 times in order to test the significance of the difference in performance scores.

## 5. Offline Results

### 5.1. Main Results

Table 2 shows that MECo-DGNN outperforms the comparison models in terms of recommendation effectiveness. MECo-DGNN achieves the best performance in terms of accuracy and fairness and consistently outperforms all comparison models across the three datasets. We conduct a one-tailed t-test between MECo-DGNN and the best-performing comparison model for each dataset to evaluate whether MECo-DGNN provides significant improvement over available benchmarking options. For all results, MECo-DGNN's score outperforms the benchmark methods. For all metrics but NDGC, the difference is significantly better than the best-performing comparison model. These results validate MECo-DGNN's effectiveness via learning the Matthew effect's dynamic evolution trend and controlling the Gini coefficient.

**Table 2** Performance Comparison between MECo-DGNN and Comparison Models on Proprietary Data

| Model | Recall | Precision | NDCG | APT | Gini |
|-------|--------|-----------|------|-----|------|
| MF | 10.14 | 1.20 | 4.52 | 5.20 | 88.48 |
| NeuralCF | 10.08 | 1.19 | 4.53 | 4.65 | 88.83 |
| DyGCN | 11.55 | 1.35 | 4.99 | 9.05 | 85.85 |
| BPRMF | 10.65 | 1.23 | 4.81 | 4.87 | 89.08 |
| +PC | 11.60 | 1.31 | 5.30 | 5.44 | 86.08 |
| +PDA | 11.32 | 1.34 | 5.14 | 5.52 | 85.42 |
| LightGCN | 10.55 | 1.23 | 5.47 | 4.66 | 88.71 |
| +PC | 11.45 | 1.32 | 5.39 | 5.32 | 86.53 |
| +PDA | 11.17 | 1.25 | 5.34 | 6.31 | 86.65 |
| SASRec | 9.86 | 1.12 | 5.19 | 4.29 | 88.64 |
| +PC | 11.48 | 1.30 | 5.26 | 5.20 | 86.49 |
| +PDA | 11.32 | 1.28 | 5.41 | 4.66 | 88.63 |
| RL | 12.22 | 1.30 | 5.23 | 6.53 | 87.91 |
| **MECo-DGNN** | **12.39** | **1.43** | **5.54** | **11.94** | **83.41** |

*Notes:* All metrics shown as percentages for readability. Best performing model is in bold. Best performing comparison model is underlined. For all metrics except **Gini**, a higher score means better performance. One-tailed t-test between our model and best performing comparison model. **$p < 0.05$.

Figure 3 shows the result of our ablation study. If we exclude stage degree regularization (i.e., $\alpha = 0$), both diversity and precision degrade. Excluding the Matthew regularization (i.e., $\beta = 0$) improves prediction performance at the expense of diversity. If we enforce perfect equality (i.e., $Gini_{ideal} = 0$), we see the most diverse results with higher APT scores and lower Gini coefficient
scores. However, the predictive performance suffers greatly. Lastly, a static implementation of our framework, without the dynamic representation, degrades both prediction performance and diversity. Although it has typically been assumed that there is a diversity-performance tradeoff in machine learning, our results are consistent with recent prior work disputing that claim (Wick and Tristan 2019, Peng et al. 2024). Our results here bolster this new result regarding diversity and performance, specifically with regard to the fact that the operationalization of diversity, and its relevance to the problem, is critical when conducting evaluation. Across our ablations, we see many cases where performance is improved compared to the best performing benchmark (dotted lines in Figure 3). By adding the Matthew effect regularization and operational control of $Gini_{target}$, we allow managers to control the balance between diversity and accuracy while maintaining high performance.

**Figure 3** Ablation Analysis of MECo-DGNN Model Components. For all metrics except Gini, a higher score means better performance. Dotted lines represent the best benchmark model result from Table 2.

[Figure showing 5 scatter plots for Recall, Precision, NDCG, APT, and Gini metrics across different model variants: MECo-DGNN, MECo-GNN, α=0, β=0, and Ginitarget=0]

### 5.2. Empirical Extensions: Offline Evaluation with Public Data

We further evaluate the framework of MECo-DGNN via a thorough set of analyses across three widely adopted public data sets: MovieLens 25M, Amazon Movie, and Douban Book. As shown in Figure 4, we find that MECo-DGNN outperforms the best-performing comparison model, with the exception of NDCG for Douban Book. To ensure the validity of our proposed control modules, we present results from ablation experiments analyzing the impact of hyperparameters and the ideal Gini coefficient on MECo-DGNN's performance in Figure 5. As in our partner platform evaluation, MECo-DGNN best balances accuracy and diversity across the ablations. We refer the readers to Appendix EC.4 for details about the datasets and experimental setup.
**Figure 4** Comparison of MECo-DGNNwith benchmark methods on public data.

[Figure showing dot plots comparing MECo-DGNN with benchmark methods (MF, NeuralCF, DyGCN, BPRMF, BPRMF_PC, BPRMF_PDA, SASRec, SASRec_PC, SASRec_PDA, LightGCN, LightGCN_PC, LightGCN_PDA, MECo-DGNN) across three datasets (Amazon, Douban, MovieLens) and five metrics (NDCG, P, R, APT, Gini)]

**Figure 5** Ablation of MECo-DGNNwith benchmark methods on public data.

[Figure showing dot plots for ablation analysis across three datasets (Amazon, Douban, MovieLens) and five metrics (APT, Gini, NDCG, P, R) comparing MECo-DGNN variants: MECo-DGNN, MECo-DGNN_MECo-GNN, MECo-DGNN_GiniIdeal0, MECo-DGNN_noBeta, MECo-DGNN_noAlpha]
**Figure 4** Comparison of MECo-DGNNwith benchmark methods on public data.

[Figure showing dot plots comparing MECo-DGNN with benchmark methods (MF, NeuralCF, DyGCN, BPRMF, BPRMF_PC, BPRMF_PDA, SASRec, SASRec_PC, SASRec_PDA, LightGCN, LightGCN_PC, LightGCN_PDA, MECo-DGNN) across three datasets (Amazon, Douban, MovieLens) and five metrics (NDCG, P, R, APT, Gini)]

**Figure 5** Ablation of MECo-DGNNwith benchmark methods on public data.

[Figure showing dot plots for ablation analysis across three datasets (Amazon, Douban, MovieLens) and five metrics (APT, Gini, NDCG, P, R) comparing MECo-DGNN variants: MECo-DGNN, MECo-DGNN_MECo-GNN, MECo-DGNN_GiniIdeal0, MECo-DGNN_noBeta, MECo-DGNN_noAlpha]
[Note: This page contains continuation of figures from previous pages]
effective algorithms for modeling the dynamic Matthew effect and identifying an optimal Lorenz curve approximation equation that aligns with recommender systems' item distribution.

Our work has several implications for practitioners and researchers. For practitioners, using a 12-month field experiment on a large on-demand service platform, we credibly establish the need to improve item (provider) diversity as an objective of recommender systems. Prior work has mostly demonstrated this notion in a descriptive or theoretical fashion. We also provide a framework to control the Matthew effect within recommender systems by leveraging the dynamic nature of the problem during model training. By allowing firms to determine their own ideal Gini coefficient, we provide a mechanism for operational decision makers to balance diversity and performance with their business needs. Lastly, we present evidence that MECo-DGNN improves on key operational priorities and can drive increased revenue for firms.

For researchers, we extend the complex systems approach that has been studied in the context of recommender systems to diversity considerations. While the approach has been shown to be useful for improving recommender system performance, it has not been applied to mitigate biases related to the Matthew effect. Here, we show that by modeling the Matthew effect within recommender systems, we can better balance diversity and performance in the pursuit of responsible recommender systems by design (Abbasi et al. 2018, Zhu 2022). We furthermore introduce a novel component that balances network structure and node degree to mitigate the "rich get richer" nature of the Matthew effect. Because of our target Gini coefficient and dynamic representations, diversity corrections via retraining will be less common and can be incorporated into standard retraining pipelines.

Our work has limitations that can inform future research. Although pre-specifying $Gini_{target}$ offers input for operational considerations of key performance indicators, further study of the considerations and how they affect $Gini_{target}$ from a socio-technical perspective, as well the impact on predictive, diversity, and operational metrics, is needed. In addition, model extensions incorporating additional user and/or item information such as product categories, free-text descriptions, purchase history, and updated preferences in learned network embeddings could inform representational richness and subsequent performance (Dzyabura and Hauser 2019). Further, competing incentives when firm revenue is shared with market participants can affect recommendation strategies and warrant further analysis (Qian and Jain 2024).

Our proposed method relies simply on user-item interaction data, which is a form of intrinsic feedback. It is, therefore, cost-efficient from a modeling standpoint; specifically, we do not require user or item features. Recommender systems can also use extrinsic feedback (e.g., user ratings) and additional content/context as features (Unger et al. 2016, 2020, Adomavicius et al. 2021, Bauman and Tuzhilin 2022). Our public benchmarking includes extrinsic feedback in order to confirm that our framework can handle extrinsic feedback (see Appendix Table D1). In addition, recent work
on GNNs has looked to incorporate so-called heterogeneous nodes that include feature vectors. MECo-DGNN can be extended to incorporate features when learning node embeddings; this is left for future work.

## References

Abbasi A, Li J, Clifford G, Taylor H (2018) Make "fairness by design" part of machine learning. *Harvard Business Review* 1.

Abbasi A, Parsons J, Pant G, Sheng ORL, Sarker S (2024) Pathways for design research on artificial intelligence. *Information Systems Research* 35(2):441-459.

Abdollahpouri H (2020) *Popularity bias in recommendation: A multi-stakeholder perspective*. Ph.D. thesis, University of Colorado at Boulder.

Abdollahpouri H, Burke R, Mobasher B (2017) Controlling popularity bias in learning-to-rank recommendation. *Proceedings of the eleventh ACM conference on recommender systems*, 42-46.

Abdollahpouri H, Mansoury M, Burke R, Mobasher B (2019) The unfairness of popularity bias in recommendation. *RecSys Workshop on Recommendation in Multistakeholder Environments (RMSE)*.

Adomavicius G, Bauman K, Tuzhilin A, Unger M (2021) Context-aware recommender systems: From foundations to recent developments. *Recommender systems handbook*, 211-250 (Springer).

Adomavicius G, Tuzhilin A (2005) Toward the next generation of recommender systems: a survey of the state-of-the-art and possible extensions. *IEEE Transactions on Knowledge and Data Engineering* 17(6):734-749.

Anderson A, Maystre L, Anderson I, Mehrotra R, Lalmas M (2020) Algorithmic effects on the diversity of consumption on Spotify. *Proceedings of The Web Conference 2020*, 2155-2165, WWW '20 (New York, NY, USA: Association for Computing Machinery), ISBN 9781450370233.

Armstrong M (2006) Competition in two-sided markets. *The RAND journal of economics* 37(3):668-691.

Arthur JK, Doh RF, Mantey EA, Osei-Kwakye J (2019) Complex network based recommender system. *International Journal of Computer Applications* 975:8887.

Bastani H, Bayati M, Khosravi K (2021) Mostly exploration-free algorithms for contextual bandits. *Management Science* 67(3):1329-1349.

Bauman K, Tuzhilin A (2022) Know thy context: Parsing contextual information from user reviews for recommendation purposes. *Information Systems Research* 33(1):179-202.

Berman R, Katona Z (2020) Curation algorithms and filter bubbles in social networks. *Marketing Science* 39(2):296-316.

Berman R, Zhao H, Zhu Y (2024) Strategic design of recommendation algorithms. *SSRN 4301489*.

Bertsimas D, Farias VF, Trichakis N (2011) The price of fairness. *Operations Research* 59(1):17-31.
Besbes O, Kanoria Y, Kumar A (2024) The fault in our recommendations: On the perils of optimizing the measurable. *Proceedings of the 18th ACM Conference on Recommender Systems*, 200-208.

Biega AJ, Gummadi KP, Weikum G (2018) Equity of attention: Amortizing individual fairness in rankings. *The 41st international ACM SIGIR conference on research & development in information retrieval*, 405-414.

Bobadilla J, Ortega F, Hernando A, Gutiérrez A (2013) Recommender systems survey. *Knowledge-Based Systems* 46:109-132, ISSN 0950-7051.

Boccaletti S, Latora V, Moreno Y, Chavez M, Hwang DU (2006) Complex networks: Structure and dynamics. *Physics reports* 424(4-5):175-308.

Brynjolfsson E, Chen L, Gao X (2025) Gains from product variety: Evidence from a large digital platform. *Information Systems Research*.

Brynjolfsson E, Hu Y, Simester D (2011) Goodbye pareto principle, hello long tail: The effect of search costs on the concentration of product sales. *Management Science* 57(8):1373-1386.

Chen J, Dong H, Wang X, Feng F, Wang M, He X (2023) Bias and debias in recommender system: A survey and future directions. *ACM Transactions on Information Systems* 41(3):1-39.

Chotikapanich D (1993) A comparison of alternative functional forms for the Lorenz curve. *Economics Letters* 41(2):129-138.

Cinelli M, Morales GDF, Galeazzi A, Quattrociocchi W, Starnini M (2021) The echo chamber effect on social media. *Proceedings of the National Academy of Sciences* 118(9):e2023301118.

Dhillon PS, Aral S (2021) Modeling dynamic user interests: A neural matrix factorization approach. *Marketing Science* 40(6):1059-1080, URL http://dx.doi.org/10.1287/mksc.2021.1293.

Donahue K, Kleinberg J (2020) Fairness and utilization in allocating resources with uncertain demand. *Proceedings of the 2020 conference on fairness, accountability, and transparency*, 658-668.

Donnelly R, Kanodia A, Morozov I (2024) Welfare effects of personalized rankings. *Marketing Science* 43(1):92-113.

Dzyabura D, Hauser JR (2019) Recommending products when consumers learn their preference weights. *Marketing Science* 38(3):417-441.

Farias VF, Li AA (2019) Learning preferences with side information. *Management Science* 65(7):3131-3149.

Fleder D, Hosanagar K (2009) Blockbuster culture's next rise or fall: The impact of recommender systems on sales diversity. *Management Science* 55(5):697-712.

Gao C, Zheng Y, Li N, Li Y, Qin Y, Piao J, Quan Y, Chang J, Jin D, He X, et al. (2023) A survey of graph neural networks for recommender systems: Challenges, methods, and directions. *ACM Transactions on Recommender Systems* 1(1):1-51.
Gers FA, Schmidhuber J, Cummins F (2000) Learning to forget: Continual prediction with LSTM. *Neural computation* 12(10):2451-2471.

Ghose A, Ipeirotis PG, Li B (2014) Examining the impact of ranking on consumer behavior and search engine revenue. *Management Science* 60(7):1632-1654.

Goel V, Sahnan D, Dutta S, Bandhakavi A, Chakraborty T (2023) Haters ride on echo chambers to escalate hate speech diffusion. *PNAS nexus* 2(3):pgad041.

Greenwood S, Chiniah S, Garg N (2024) User-item fairness tradeoffs in recommendations. *Advances in Neural Information Processing Systems* 37:114236-114288.

Hagiu A, Wright J (2024) Optimal discoverability on platforms. *Management Science* 70(11):7770-7790.

He X, Deng K, Wang X, Li Y, Zhang Y, Wang M (2020) LightGCN: Simplifying and powering graph convolution network for recommendation. *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, 639-648.

He X, Liao L, Zhang H, Nie L, Hu X, Chua TS (2017) Neural collaborative filtering. *Proceedings of the 26th international conference on world wide web*, 173-182.

Hochreiter S, Schmidhuber J (1997) Long short-term memory. *Neural computation* 9(8):1735-1780.

Hosanagar K, Fleder D, Lee D, Buja A (2014) Will the global village fracture into tribes? Recommender systems and their effects on consumer fragmentation. *Management Science* 60(4):805-823.

Hu S, Zhang J, Zhu Y (2024) Beyond zero: Jump-starting sales with a recommender system for missing-by-choice data.

Jannach D, Jugovac M (2019) Measuring the business value of recommender systems. *ACM Transactions on Management Information Systems (TMIS)* 10(4):1-23.

Kang WC, McAuley J (2018) Self-attentive sequential recommendation. *2018 IEEE international conference on data mining (ICDM)*, 197-206 (IEEE).

Kitchens B, Claggett JL, Abbasi A (2024) Timely, granular, and actionable: Designing a social listening platform for public health 3.0. *MIS Quarterly* 48(3).

Kumar A, Hosanagar K (2019) Measuring the value of recommendation links on product demand. *Information Systems Research* 30(3):819-838.

Kumar M, Silva P, Singh A, Varmaraja A (2024) Inclusive recommendations and user engagement: Experimental evidence from Pinterest. *Proceedings of the 25th ACM Conference on Economics and Computation*, 1189-1191, EC '24 (New York, NY, USA: Association for Computing Machinery).

Lee D, Hosanagar K (2019) How do recommender systems affect sales diversity? A cross-category investigation via randomized field experiment. *Information Systems Research* 30(1):239-259.

Lee D, Hosanagar K (2021) How do product attributes and reviews moderate the impact of recommender systems through purchase stages? *Management Science* 67(1):524-546.
Levy R (2021) Social media, news consumption, and polarization: Evidence from a field experiment. *American Economic Review* 111(3):831-70.

Li P, Tuzhilin A (2024) When variety seeking meets unexpectedness: Incorporating variety-seeking behaviors into design of unexpected recommender systems. *Information Systems Research* 35(3):1257-1273.

Li Y, Liu J, Ren J (2019a) Social recommendation model based on user interaction in complex social networks. *PloS one* 14(7):e0218957.

Li Y, Meng C, Shahabi C, Liu Y (2019b) Structure-informed graph auto-encoder for relational inference and simulation. *ICML Workshop on Learning and Reasoning with Graph-Structured Data*, volume 8, 2.

Lillicrap TP, Hunt JJ, Pritzel A, Heess N, Erez T, Tassa Y, Silver D, Wierstra D (2016) Continuous control with deep reinforcement learning. Bengio Y, LeCun Y, eds., *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*.

Lin X, Wang W, Li Y, Yang S, Feng F, Wei Y, Chua TS (2024) Data-efficient fine-tuning for LLM-based recommendation. *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 365-374.

Liu F, Tang R, Li X, Zhang W, Ye Y, Chen H, Guo H, Zhang Y (2018) Deep reinforcement learning based recommendation with explicit user-item interactions modeling. *arXiv preprint arXiv:1810.12027*.

Liu J, Wang G, Zhao H, Lu M, Huang L, Chen G (2025) Beyond complements and substitutes: A graph neural network approach for collaborative retail sales forecasting. *Information Systems Research*.

Liu X (2023) Dynamic coupon targeting using batch deep reinforcement learning: An application to livestream shopping. *Marketing Science* 42(4):637-658.

Ma T, Hu Y, Lu Y, Bhattacharyya S (2024) Customer engagement prediction on social media: A graph neural network method. *Information Systems Research*.

Malgonde O, Zhang H, Padmanabhan B, Limayem M (2020) Taming complexity in search matching: Two-sided recommender systems on digital platforms. *MIS Quarterly* 44(1):49-84.

Manessi F, Rozza A, Manzo M (2020) Dynamic graph convolutional networks. *Pattern Recognition* 97:107000.

Mansoury M, Abdollahpouri H, Pechenizkiy M, Mobasher B, Burke R (2020) Feedback loop and bias amplification in recommender systems. *Proceedings of the 29th ACM international conference on information & knowledge management*, 2145-2148.

Milgram S (1967) The small world problem. *Psychology today* 2(1):60-67.

Mladenov M, Creager E, Ben-Porat O, Swersky K, Zemel R, Boutilier C (2020) Optimizing long-term social welfare in recommender systems: A constrained matching approach. *International Conference on Machine Learning*, 6987-6998 (PMLR).

Molaie MM, Lee W (2022) Economic corollaries of personalized recommendations. *Journal of Retailing and Consumer Services* 68:103003.
Oestreicher-Singer G, Sundararajan A (2012) Recommendation networks and the long tail of electronic commerce. *MIS Quarterly* 36(1):65-84.

Padmanabhan B, Tuzhilin A (1999) Unexpectedness as a measure of interestingness in knowledge discovery. *Decision Support Systems* 27(3):303-318.

Park Y, Bang Y, Ahn JH (2020) How does the mobile channel reshape the sales distribution in e-commerce? *Information Systems Research* 31(4):1164-1182.

Park YJ, Tuzhilin A (2008) The long tail of recommender systems and how to leverage it. *Proceedings of the 2008 ACM conference on Recommender systems*, 11-18.

Paul S, Shankar S (2020) An alternative single parameter functional form for Lorenz curve. *Empirical Economics* 59:1393-1402.

Peng K, Raghavan M, Pierson E, Kleinberg J, Garg N (2024) Reconciling the accuracy-diversity trade-off in recommendations. *Proceedings of the ACM Web Conference 2024*, 1318-1329, WWW '24 (New York, NY, USA: Association for Computing Machinery).

Prawesh S, Padmanabhan B (2021) A complex systems perspective of news recommender systems: Guiding emergent outcomes with feedback models. *Plos one* 16(1):e0245096.

Qian K, Jain S (2024) Digital content creation: An analysis of the impact of recommendation systems. *Management Science*.

Rafieian O, Kapoor A, Sharma A (2024) Multiobjective personalization of marketing interventions. *Marketing Science*.

Rakova B, Yang J, Cramer H, Chowdhury R (2021) Where responsible AI meets reality: Practitioner perspectives on enablers for shifting organizational practices. *Proceedings of the ACM on Human-Computer Interaction* 5(CSCW1):1-23.

Ren X, Wei W, Xia L, Su L, Cheng S, Wang J, Yin D, Huang C (2024) Representation learning with large language models for recommendation. *Proceedings of the ACM on Web Conference 2024*, 3464-3475.

Rendle S, Freudenthaler C, Gantner Z, Schmidt-Thieme L (2009) BPR: Bayesian personalized ranking from implicit feedback. *Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence*, 452-461.

Rigney D (2010) *The Matthew effect: How advantage begets further advantage* (Columbia University Press).

Rochet JC, Tirole J (2003) Platform competition in two-sided markets. *Journal of the European economic association* 1(4):990-1029.

Santos FP, Lelkes Y, Levin SA (2021) Link recommendation algorithms and dynamics of polarization in online social networks. *Proceedings of the National Academy of Sciences* 118(50):e2102141118.

Saveski M, Mantrach A (2014) Item cold-start recommendations: learning local collective embeddings. *Proceedings of the 8th ACM Conference on Recommender systems*, 89-96.
Shi Z, Raghu T (2020) An economic analysis of product recommendation in the presence of quality and taste-match heterogeneity. *Information Systems Research* 31(2):399-411.

Shulman JD, Toubia O, Saddler R (2023) Marketing's role in the evolving discipline of product management. *Marketing Science* 42(1):1-5.

Singh A, Joachims T (2018) Fairness of exposure in rankings. *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, 2219-2228.

Song Y, Sahoo N, Ofek E (2019) When and how to diversify—a multicategory utility model for personalized content recommendation. *Management Science* 65(8):3737-3757.

Steck H (2011) Item popularity and recommendation accuracy. *Proceedings of the fifth ACM conference on Recommender systems*, 125-132.

Steck H (2019) Collaborative filtering via high-dimensional regression. *arXiv preprint arXiv:1904.13033*.

Strogatz SH (2001) Exploring complex networks. *Nature* 410(6825):268-276.

Su Z, Lin Z, Ai J, Li H (2021) Rating prediction in recommender systems based on user behavior probability and complex network modeling. *IEEE Access* 9:30739-30749.

Tsamados A, Floridi L, Taddeo M (2024) Human control of AI systems: From supervision to teaming. *AI and Ethics* 1-14.

Tu K, Cui P, Wang X, Yu PS, Zhu W (2018) Deep recursive network embedding with regular equivalence. *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, 2357-2366.

Unger M, Bar A, Shapira B, Rokach L (2016) Towards latent context-aware recommendation systems. *Knowledge-Based Systems* 104:165-178.

Unger M, Tuzhilin A, Livne A (2020) Context-aware recommendations based on deep learning frameworks. *ACM Transactions on Management Information Systems (TMIS)* 11(2):1-15.

Wan X, Kumar A, Li X (2024) How do product recommendations help consumers search? Evidence from a field experiment. *Management Science* 70(9):5776-5794.

Wang H, Wang Z, Zhang W (2018) Quantitative analysis of Matthew effect and sparsity problem of recommender systems. *2018 IEEE 3rd International Conference on Cloud Computing and Big Data Analysis (ICCCBDA)*, 78-82 (IEEE).

Wang L, Joachims T (2021) User fairness, item fairness, and diversity for rankings in two-sided markets. *Proceedings of the 2021 ACM SIGIR international conference on theory of information retrieval*, 23-41.

Wang W, Li B, Luo X, Wang X (2023a) Deep reinforcement learning for sequential targeting. *Management Science* 69(9):5439-5460.

Wang X, He X, Wang M, Feng F, Chua TS (2019) Neural graph collaborative filtering. *Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval*, 165-174.
Wang Y, Ma W, Zhang M, Liu Y, Ma S (2023b) A survey on the fairness of recommender systems. *ACM Transactions on Information Systems* 41(3):1-43.

Wang Y, Tao L, Zhang XX (2025) Recommending for a multi-sided marketplace: A multi-objective hierarchical approach. *Marketing Science* 44(1):1-29.

Wei T, Feng F, Chen J, Wu Z, Yi J, He X (2021) Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 1791-1800.

Wick M, Tristan JB (2019) Unlocking fairness: a trade-off revisited. *Advances in neural information processing systems* 32.

Wire P (2023) Recommendation Engine Market to Reach $43.8 Billion, Globally, by 2031 at 32.1

Wu S, Sun F, Zhang W, Xie X, Cui B (2022) Graph neural networks in recommender systems: A survey. *ACM Computing Surveys* 55(5):1-37.

Wu Y, Cao J, Xu G (2023) Fairness in recommender systems: Evaluation approaches and assurance strategies. *ACM Transactions on Knowledge Discovery from Data* 18(1):1-37.

Wu Z, Pan S, Chen F, Long G, Zhang C, Philip SY (2020) A comprehensive survey on graph neural networks. *IEEE transactions on neural networks and learning systems* 32(1):4-24.

Xu K, Hu W, Leskovec J, Jegelka S (2019) How powerful are graph neural networks? *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019* (OpenReview.net).

Yin K, Fang X, Chen B, Sheng ORL (2023) Diversity preference-aware link recommendation for online social networks. *Information Systems Research* 34(4):1398-1414.

Zangerle E, Bauer C (2022) Evaluating recommender systems: Survey and framework. *ACM Computing Surveys* 55(8):1-38.

Zeng Z, Zhang Z, Zhang D, Chan T (2024) The impact of recommender systems on content consumption and production: Evidence from field experiments and structural modeling. *Available at SSRN*.

Zhang F, Shi S, Zhu Y, Chen B, Cen Y, Yu J, Chen Y, Wang L, Zhao Q, Cheng Y, et al. (2024) OAG-Bench: A human-curated benchmark for academic graph mining. *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 6214-6225.

Zhang J, Adomavicius G, Gupta A, Ketter W (2020) Consumption and performance: Understanding longitudinal dynamics of recommender systems via an agent-based simulation framework. *Information Systems Research* 31(1):76-101.

Zhang W, Guo X, Wang W, Tian Q, Pan L, Jiao P (2021a) Role-based network embedding via structural features reconstruction with degree-regularized constraint. *Knowledge-Based Systems* 218:106872.
Zhang Y, Feng F, He X, Wei T, Song C, Ling G, Zhang Y (2021b) Causal intervention for leveraging popularity bias in recommendation. *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 11-20.

Zhao Y, Wang Y, Liu Y, Cheng X, Aggarwal CC, Derr T (2025) Fairness and Diversity in Recommender Systems: A Survey. *ACM Trans. Intell. Syst. Technol.* 16(1):2:1-2:28.

Zhou T, Kuscsik Z, Liu JG, Medo M, Wakeling JR, Zhang YC (2010) Solving the apparent diversity-accuracy dilemma of recommender systems. *Proceedings of the National Academy of Sciences* 107(10):4511-4515.

Zhu Y, Xie R, Zhuang F, Ge K, Sun Y, Zhang X, Lin L, Cao J (2021a) Learning to warm up cold item embeddings for cold-start recommendation with meta scaling and shifting networks. *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 1167-1176.

Zhu Z (2022) *Toward Responsible Recommender Systems*. Ph.D. thesis, Texas A&M University.

Zhu Z, He Y, Zhao X, Caverlee J (2021b) Popularity bias in dynamic recommendation. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2439-2449.

Zhu Z, He Y, Zhao X, Zhang Y, Wang J, Caverlee J (2021c) Popularity-opportunity bias in collaborative filtering. *Proceedings of the 14th ACM International Conference on Web Search and Data Mining*, 85-93.

Zou T, Wu Y, Sarvary M (2024) Designing recommendation systems on content platforms: Trading off quality and variety. *Available at SSRN*.
# Web Appendix: Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach

## EC.1. The Prevalence of the Matthew Effect across Platforms

To assess the Matthew effect's prevalence across platforms, we assemble a large-scale dataset with 1,149,610 users and 28,652,779 user-item interactions from eight platforms (for an overview, see Table A1). We select these platforms because of their size, broad industry coverage, and prevalence in the literature.

Each dataset includes features such as ratings, transactions, reviews, item details, user profiles, tags/labels, and dates. We partition the data by time (i.e., year/week/day based on the dataset) into five groups, T0-T4. To visualize the Lorenz curve, we plot the cumulative percentage of interactions for each item from least popular to most popular at each timestep (Figure A1). In Douban's book-rating platform, for example, each time period includes approximately 180,000 user-book rating interactions. The Douban book Lorenz curve (Figure A1, Douban) shows increasing inequality over time, as the distance between the curve and line of equality grows at each step. That is, item distribution becomes increasingly unequal with time. We observe similar trends across all eight considered platforms.

In summary, we document a robust Matthew effect across eight popular platforms in search, retail, media, and entertainment, that is, an unequal distribution of items across popular and niche categories. In addition, as recommender systems receive data on increasingly rich interactions among users, items, and themselves over time, they exacerbate the Matthew effect, with the distribution of items becoming increasingly unequal. Interestingly, by taking a comprehensive approach we note the following two items. First, the Matthew effect is in fact *dynamic*. Over time, the disparity gets worse. Second, the severity of the Matthew effect is *domain-dependent*. Therefore, managerial considerations regarding how much control is required is needed.

**Table A1** Summary of platform characteristics

| Platform name | Interactions | Users | Items | Edge | Stage unit | Product categories | Business model | Establishment year |
|---------------|--------------|-------|-------|------|------------|-------------------|----------------|-------------------|
| **Amazon** | 578,187 | 134,867 | 43,603 | Rating | year | Products and services | Selling Products/services | 1994 |
| **Douban** | 894,206 | 35,701 | 176,804 | Rating | year | Entertainment | Information platform | 2005 |
| **MovieLens** | 3,217,304 | 267,741 | 186,622 | Rating | year | Entertainment | Information platform | 1997 |
| **Yelp** | 803,577 | 15,672 | 63,811 | Rating | year | Travel/restaurants | Information platform | 2004 |
| **Microsoft** | 15,021,218 | 311,026 | 70,132 | Rating | week | News | Information platform | 1975 |
| **Food.com** | 120,612 | 10,312 | 57,159 | Rating | year | Recipes | Information platform | 2012 |
| **Alibaba** | 710,241 | 320,483 | 172,135 | Click | day | Products and services | Selling Products/services | 1999 |
| **Last.fm** | 6,910,834 | 1,923 | 107,296 | Listen | year | Entertainment | Information platform | 2003 |

Specifically, 120,000 new merchants were randomly assigned to one of two conditions (60,000 in each condition) from August 16, 2022 to August 1, 2023 (about 12 months). Once a merchant was
**Figure A1** Lorenz curves for various recommender systems datasets (a-h). Over time, the distance between the Lorenz curve and line of equality increases, indicating increased exposure disparities.

[Figure showing 8 Lorenz curve plots:
(a) MovieLens
(b) Douban
(c) Yelp
(d) Amazon
(e) Alibaba
(f) Food.com
(g) Last.fm
(h) Microsoft

Each plot shows Cumulative % of Interactions vs Cumulative % of Items with curves for T0, T1, T2, T3, T4 time periods, demonstrating increasing inequality over time.]

assigned to a condition, the merchant stayed in the same condition until the experiment ended. The manipulation is that the search team allocated **Position 7/Position 8** to those merchants in the treatment condition: that is, the ranking algorithm boosted the original positions of the treated merchants to **Position 7/Position 8** if they were ranked lower than Position 8, and it maintained their original positions otherwise. Given that customers on average view 20-50 recommendations on the page, the rationale of using **Position 7/Position 8** was to ensure traffic support for new merchants while minimizing potential disruptions to the platform's regular operations and customer
satisfaction.[^16] As a result, we expect that this exogenous change would increase the likelihood of treated merchants gaining additional exposure compared to their control counterparts.

The search team granted us access to the performance insights dashboard, which reports aggregate experiment-level results. Although we do not observe merchant-level characteristics (e.g., size, tenure) nor performance on any specific day, we observe several performance indicators during the entire experiment: (a) total number of page views, (b) activity level, (c) average daily sales, (d) proportion of merchants renewing their contract with the platform, and (e) overall ratings. These indicators are informative because they capture the experimental variation tied to the focal treatment, merchant engagement, top-line growth, retention, and reputation, respectively. In addition, we observe the treatment effect estimates in the form of *lift*: the incremental change in the mean of the indicator among treated merchants ($\overline{Y_1}$) relative to control merchants ($\overline{Y_0}$) as a percentage of the mean of the indicator among control merchants ($\frac{\overline{Y_1} - \overline{Y_0}}{\overline{Y_0}}$).

Table A1 presents the results. First, we verify the success of the manipulation by checking the experimental variation induced by the treatment. Panel A shows that the number of treated merchants' page views is 120% higher than that of their control counterparts. In line with the treatment intent, this result confirms that increased exposure of new merchants in recommendations leads to an increase in the number of page views (manipulation check). In Panel B, we show that on average, relative to control counterparts, treated merchants were 6.6% more active, generated 6.8% more sales, were 12% more likely to renew the contract with the platform, and were rated 1.9% higher over the experiment period.

Collectively, these results suggest that boosting new merchants' exposure leads to an increase in their engagement, growth, and retention without compromising their reputation over the period of 12 months.

**Table A1** Causal Effects of Boosting Merchant Exposure

| Merchant-level Performance Indicator | Lift |
|--------------------------------------|------|
| **A. Experimental Variation** | |
| Merchant exposure page views | 120%*** |
| **B. Average Treatment Effects** | |
| Activity level | 6.6%** |
| Average daily sales | 6.8%*** |
| Renewal rate | 12.0%*** |
| Merchant ratings | 1.9%*** |

*Notes:* N=120,000. Lift refers to the incremental change in the mean of the indicator among treated merchants relative to control merchants as a percentage of the mean of the indicator among control merchants. Indicators were computed over the entire experiment. ***p < 0.01; **p < 0.05.

[^16]: Positions 7 and 8 were identified by the platform based on internal experimentation. In the treatment, a merchant was randomly assigned to Position 7 or 8.
## EC.2. LSTM

LSTMs are commonly used for sequential modeling that incorporates long-term correlations (Hochreiter and Schmidhuber 1997, Gers et al. 2000); this is key to capturing the evolutionary pattern of the dynamic network. LSTMs model input, output, forget, and cell gates ($i_t$, $o_t$, $f_t$, and $c_t$, respectively) via trainable weights $W$ and biases $b$. We calculate the hidden state of the LSTM network by feeding in the sequence of historical network snapshots as follows.

$$
X = [x_1, x_2, \ldots, x_T] {\text{(EC.1)}}
$$

$$
i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i) {\text{(EC.2)}}
$$

$$
f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f) {\text{(EC.3)}}
$$

$$
c_t = f_t c_{t-1} + i_t \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c) {\text{(EC.4)}}
$$

$$
o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o) {\text{(EC.5)}}
$$

$$
h_t = o_t \tanh(c_t) {\text{(EC.6)}}
$$

## EC.3. Evaluation Metrics

Normalized discounted cumulative gain (NDCG) is a metric used to evaluate the relevance of recommended results. The gain represents the relevance score assigned to each recommended item. The cumulative gain (CG) at position K is calculated as the sum of gains for the first K recommended items. To account for the position of each item, the discounted cumulative gain (DCG) assigns weights to relevance scores. Items at the top receive higher weights, while those at the bottom receive lower weights. NDCG is obtained by dividing DCG by a normalization factor. The denominator represents the ideal DCG score achieved when the most relevant items are recommended first. Expressed formally:

$$
\text{CG@k} = \sum_i^k Gain_i, {\text{(EC.7)}}
$$

$$
\text{DCG@k} = \sum_i^k \frac{Gain_i}{log_2(i+1)}, {\text{(EC.8)}}
$$

$$
\text{IDCG@k} = \sum_{i=1}^{K^{idea}} \frac{1}{U} Gain_i^{ideal} log_2(i+1), {\text{(EC.9)}}
$$

$$
\text{NDCG@k} = \frac{DCG@k}{IDCG@k}. {\text{(EC.10)}}
$$

Recommendation involves retrieving items predicted to be "good" (i.e., items users eventually click on or purchase). Recall is a metric that gauges completeness level and determines the proportion of relevant items that are retrieved out of all the relevant items that exist. Precision determines the proportion of relevant items retrieved out of all items. Recall and precision are defined as:

$$
\text{Recall} = \frac{a}{a+c} {\text{(EC.11)}}
$$

$$
\text{Precision} = \frac{a}{a+b} {\text{(EC.12)}}
$$

Where $a$ denotes the count of true positive items that are originally recommended and successfully retrieved by the recommendation system, $b$ indicates the number of items labeled as recommended by the system but not effectively suggested, $c$ represents the count of items recommended but deemed irrelevant or unsuitable, and $d$ corresponds to the true negative count, referring to items accurately identified and retrieved as "not recommended."

An effective recommender system strives to optimize both recall and precision concurrently. For instance, it could suggest numerous products or items to the user, achieving extensive coverage, while remaining low on precision, being constrained by the ratio of beneficial products or items within the pool.

The diversity evaluation metric we employ is the average percentage of tail items (APT) in the recommendation lists (Abdollahpouri et al. 2017). APT measures the average proportion of less popular items that appear in the recommendations, which we define as follows:

$$
\text{APT} = \frac{1}{|U_t|} \sum_{u \in U_t} \frac{|i, i \in (L(u) \cap \sim \Phi)|}{L(u)}, {\text{(EC.13)}}
$$

where $U_t$ is the number of users in the test set, $L(u)$ is the recommendation list for user u, and $\sim \Phi$ is the set of medium-tail items. The measure tells us what percentage of items in users' recommendation lists belongs to the medium tail set.

The Gini coefficient is widely adopted as the primary metric for measuring distribution of resources across a population. The utility can be defined as either the predicted relevance for a user or the exposure for an item. A lower Gini coefficient indicates fairer recommendations, as it reflects a more equitable distribution of utility among users or items. However, like income distribution, this doesn't imply that a lower Gini coefficient is always better. Absolute equality (Gini coefficient of 0) could be detrimental to the vitality of both society and businesses. We define the Gini coefficient as:

$$
\text{Gini} = \frac{\sum_{v_x, v_y \in V} |f(v_x) - f(v_y)|}{2|V| \sum_v f(v)}. {\text{(EC.14)}}
$$

Where $f(*)$ is the utility function for individuals or groups.

## EC.4. Evaluating MECo-DGNN with Public Data

We evaluate the framework of MECo-DGNN via a thorough set of empirical analyses, including benchmark analyses against multiple state-of-the-art recommendation models across three representative public datasets. In this section, we supplement the results in Figures 4 and 5 with a description of the datasets and the experimental setup.

### EC.4.1. Datasets

We obtain three widely used datasets with varying degrees of sparsity: MovieLens 25M, Amazon Movie, and Douban Book (Table D1). The MovieLens dataset consists of 25 million ratings spanning 162,000 users and 62,000 movies. The Amazon Movie dataset contains 7,911,684 interactions between 889,172 users and 253,059 items. The Douban Book dataset is obtained from the Chinese review website Douban and consists of users' book ratings. The dataset contains 271,417,899 interactions between 1,434,209 users and 404,967 items. For our analysis, we consider all rating records as positive samples and use data collected from 2013 to 2017. The resulting date range for our analyses is 2013 to 2017; we split the data by year into five groups: T0, T1, T2, T3, and T4.[^17]

**Table D1** Benchmark Dataset Summary Statistics

| Dataset | Users | Items | Nodes | Years/Weeks |
|---------|-------|-------|-------|-------------|
| MovieLens | 162,000 | 62,000 | 25,000,000 | 24 |
| Amazon Movie | 889,176 | 253,059 | 7,911,684 | 10 |
| Douban Book | 383,033 | 89,908 | 13,506,215 | 10 |

### EC.4.2. Comparison with Representative Models

[^17]: For consistent analyses over a fixed period, we remove data falling outside our target time range.
