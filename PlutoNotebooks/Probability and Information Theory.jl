### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 02d67371-b1e3-4ac9-b4d1-cdca1c85bd18
using Distributions, Statistics

# ╔═╡ 7c8394db-271e-4d0b-a3db-b3e3a659ec0a
using Gaston; set(showable="svg")

# ╔═╡ bf7ce886-e6ac-45e7-a0f5-98e36c8974c7
using LightGraphs, GraphPlot

# ╔═╡ 7a514f20-e7cc-43da-96f6-cc4f7244166b
using PlutoUI

# ╔═╡ c7a5ab0b-aba0-4b54-abe9-702b0bad4a8e
using SimpleWeightedGraphs

# ╔═╡ ea0dc819-e860-4863-b9ea-81d12d92c4ab
md"[Reference Chapter](https://www.deeplearningbook.org/contents/prob.html)"

# ╔═╡ e84a31da-8ce7-11eb-1b6a-0f1431d8c96e
md"# Probability"

# ╔═╡ 48727d46-8cf8-11eb-25a0-73c248dab2da
md"
The probability of an event A, given B

$ P(A|B) = \frac{P(A \cap B)}{P(B)}$

The probability of B, given A

$ P(B|A) = \frac{P(B \cap A)}{P(A)}$

We can rewrite the above $ 2 $ as --

$ \frac{P(A|B)}{P(B|A)} = \frac{P(A)}{P(B)} $ 
"

# ╔═╡ af0e8170-9163-11eb-2e48-4dcc7c853090
md"**What does $P(A \cap B)$ mean?**"

# ╔═╡ dde86ab8-9163-11eb-173a-0d0fe5d297ad
md"Probability of event A _and_ event B occurring."

# ╔═╡ 51cf0da0-8cf8-11eb-13d1-53bb6bd3892a
md"
#### Question --

In 5 coin tosses, whats the chance that we get exactly 3 heads, given that we've atleast one head?
"

# ╔═╡ 5c51aa26-8cf8-11eb-0add-a1b7f41216f6
md"
So... in how many ways can I rearrange 2 squares, in a collection of 3 objects, consisting of 2 squares and a circle?

⭘ ◻ ◻, ◻ ◻ ⭘, ◻ ⭘ ◻

3 ways, without repetition

Well... what about with repetition?

6 ways(if you consider the two squares to be 2 different objects)

So how do we generalize this?

$ ^{n}P_{r} = \frac{n!}{(n-r)!}$

where $ ^{n}P_{r} $ represents the permutation where we've to rearrange r objects, in a collection of n objects, and repetition is allowed.

$ ^{n}C_{r} = \frac{n!}{r!(n-r)!}$

where `` ^{n}C_{r} `` represents the permutation where we've to rearrange r objects, in a collection of n objects, and repetition is _not_ allowed.

"

# ╔═╡ 72edf972-8cf8-11eb-2441-75b6edc68854
md"For the above question... We've $ 2^5 $ or $ 32 $ possibilities after we toss the coin 5 times, or $ 32 $ possible combination of heads and tails! Since one of the possibilities is TTTTT, and we already know from the information we have, that we've at least one head, TTTTT is not possible, so we've a total of $31$ possibilities.

Out of these $31$ possibilities, we've to find the ones with exactly three heads. Going with the above formula, we have $^{5}C_{3}$ possibilities of getting exactly three heads in five tosses, which is equal to 
"

# ╔═╡ c26eb560-9167-11eb-02d3-8b4212f378e6
binomial(5,3)

# ╔═╡ 6d6ead0a-9170-11eb-049b-2dfaf08ba40e
md"So the probability of getting exactly 3 heads is $10/31$"

# ╔═╡ 5e59cfb6-9170-11eb-0215-2390218aac63
md"The below code finds all possible results we can obtain, after flipping a coin 5 times, and stores it in variable `a`. It then removes the possibility that the result is TTTTT, as per given in the question, from `a`."

# ╔═╡ 6fd98eb6-916b-11eb-19a4-5368d9cc2b31
begin a = [prod.(collect(Iterators.product(['H', 'T'], ['H', 'T'], ['H', 'T'], ['H', 'T'], ['H', 'T'])))[i] for i in 1:2^5]
	deleteat!(a,findall(a .== "TTTTT"))
end

# ╔═╡ 980bb77e-9170-11eb-3d61-19044344aa37
md"The below code finds the number of results where we get heads exactly 3 times, and stores the outcomes in the variable `b`"

# ╔═╡ 0c0ac120-916b-11eb-1029-cb8615bca99c
b = a[[(count("H", a[i]) == 3) for i in 1:length(a)]]

# ╔═╡ 4ef0b236-9172-11eb-0813-e12589886a31
md"Counting the number of elements in `a` and `b`"

# ╔═╡ 47f31ab4-9172-11eb-1bc4-f916418a47db
length(b), length(a)

# ╔═╡ eced0ac6-9171-11eb-0f2a-cd5fa1bd818d
md"We get our answer equal to $ $(length(b)) / $(length(a)) $"

# ╔═╡ 3f168c24-9173-11eb-1cf8-7338a422ff69
md"## Random Variable"

# ╔═╡ f6b2bf60-9173-11eb-145d-2f1f47b7eb30
md"""
!!! question "Consider this example"
	I've a blue bag, and a red bag.

	The blue bag has 3 green balls and 5 yellow balls, and the red bag has 6 green balls and 4 yellow balls.

	What is the probability that when I choose a random bag and draw a ball, its a green ball?
"""

# ╔═╡ 79557c7c-9173-11eb-3c53-4fa9ad783e8a
md"""
In the above example, the color of the bag and the color of balls are _random variables_. Let's call the color of bag, B, and the color of balls, A.

We've to find out $P(A=🟢)$.

The probability that I choose a blue bag is, denoted by $P(B=$"blue"$)$, and the probability that I choose the red bag is denoted by $P(B=$"red"$)$.

Mind that the random variable _color of bag_ can have two outcomes. It can either be blue or yellow with equally likely chances with the total probability bound to be $1$. Which means, $P(B=$"blue"$)$ $= P(B=$"red"$)$ and $P(B=$"blue"$) + P(B=$"red"$) = 1$.
Which means $P(B=$"blue"$) = P(B=$"red"$) = 0.5$.

Try thinking of it this way, when an unbiased coin is tossed. Very similar to the above case, you've two expected outcomes, C = "Heads" or C = "Tails", where C is a random variable denoting the side when the coin is tossed. Here too, $P(C = $"Heads" $) = P(C = $"Tails" $) = 0.5$.

"""

# ╔═╡ 35cfa732-9238-11eb-3f47-0d921b5d6eeb
md"## Types of random variables and Joint Probability"

# ╔═╡ b4374be0-9173-11eb-09d2-cbab6657cdaf
md"""
Returning to our previous example,

$ P(A=🟢) = P(B= $ "red"$, A=🟢) + P(B=$"blue"$, A=🟢)$

Probability that the ball we chose is a green ball = (Probability that we chose the red bag, and a green ball) + (Probability that we chose the blue bag, and a green ball)

$ P(A=🟢) = \frac{1}{2} \times \frac{3}{8}  + \frac{1}{2} \times \frac{6}{10}$ 

$ P(A=🟢) = \frac{39}{80}$ or $ $(39/80) $

"""

# ╔═╡ 70e11290-923b-11eb-2365-f58a04d0f844
md"""
The probability of two or more events occuring at the same time, is called **Joint Probability**

You see it in the above example and you see it in the below question.
"""

# ╔═╡ 758a1460-8cf8-11eb-0ab7-7dfcfac203f9
md"""
!!! question "An example"
	In a particular hospital during Flu season, it was observed that 10% of the patients had Cold. Among the patients with cold, 80% had Flu and among the rest, only 5% had flu. Given that a particular patient had Flu, what is the probability that he had Cold?
"""

# ╔═╡ 806857e8-8cf8-11eb-3756-bff672168730
md"""
Probability that a patient had cold = $ 0.1 $

Probability that a patient had cold, and had flu = $ 0.1 × 0.8 $ = $ 0.08 $

Probability that a patient didn't have cold, and had flu = $ 0.9 × 0.05 $ = $ $(round((0.9*0.05); digits=3)) $

Probability that a patient has cold, given that he had flu = $\frac{Probability\;that\;a\;patient\;has\;cold\;and\;flu}{Probability\;that\;a\;patient\;has\;flu} = \frac{0.08}{0.08+0.045}$ = $ \frac{16}{25} $
"""

# ╔═╡ 7bc55826-92de-11eb-258b-31d565464c56
md"""
!!! question "Question"
	A man arrives at a bus stop at a random time (that is, with no regard for the scheduled service) to catch the next bus. Buses run every $30$ minutes without fail, hence the next bus will come any time during the next $30$ minutes with evenly distributed probability (a uniform distribution). Find the probability that a bus will come within the next $10$ minutes.
"""

# ╔═╡ a2128026-92de-11eb-102f-d924f0ceee27
md"""
In the above question, the bus can arrive any time within the next 30minutes. It has equal chances of coming at the $0.1$min or $0.01$min or $25.005$min. Every value we choose for the arrival of the bus has equal proability. For example, the probability of the bus arriving at $1$min is the same as the bus arriving at $29.909$min. Since there're infinite possibilities, for a particular given time, the probability always approaches $0$. Here, if we assume the random variable to be _the time at which the bus arrives_, its number of possible outcomes can't be counted. Such random variables are called **continous random variables**. Similarly, the random variables we encountered before this had countable number of outcomes, and were called **discrete random variables**. $𝑝(x)$ is also called the **probability density function**, and $P(x)$ is also known as the **probability mass function**. 

When it comes to **continous random variables**, we try finding the probability of the outcome falling in a particular interval, rather than a particular value as in the case of **discrete random variables**.
"""

# ╔═╡ 749c569c-92e0-11eb-14c5-8301f513cdad
md"""
In this question, since all outputs are equally likely, if we divide the time interval in three parts, the outcome would have equal chances of falling in any of the three intervals. That means, the time at which the bus arrives can be within $0$ to $10$ minutes, $10$ to $20$ minutes, or $20$ to $30$ minutes, with equally likely chances. And since the total probability is always one, following the notation from the book, 

$𝑝(0<X<10) = 𝑝(10<X<20) = 𝑝(20<X<30)$

And since the total probability is $1$, $𝑝(0<X<10) = \frac{1}{3}$
"""

# ╔═╡ e0c4991e-92e1-11eb-1cc3-31c0fae42abf
md"Similarly, if we've to choose a number from real numbers between $1$ to $10$, the random variable denoted by X, would be a **continous random variable**.

Suppose, we've to find out, what's the probaility that X is less than 3?

That would be $𝑝(X<3) = 𝑝(1<X<3) = \frac{2}{9}$. How?

Well, if we try dividing the intervals at every integer, we've 

$𝑝(1<X<2) = 𝑝(2<X<3) = 𝑝(3<X<4) = 𝑝(4<X<5) = 𝑝(5<X<6) = 𝑝(6<X<7) = 𝑝(7<X<8) = 𝑝(8<X<9) = 𝑝(9<X<10) = \frac{1}{9}$

and $𝑝(X<3) = 𝑝(1<X<3) = 𝑝(1<X<2) + 𝑝(2<X<3)$."

# ╔═╡ 1c95027c-930e-11eb-2a05-8322da402360
a1 = rand(0:0.00001:30, 1000000)

# ╔═╡ a6073126-930c-11eb-2785-79a71976e4d5
b1 = a1[a1 .<= 10]

# ╔═╡ f9c89c3c-930c-11eb-278d-9bf0ee036c67
length(b1)/length(a1)

# ╔═╡ 16b2a794-9311-11eb-3604-f1efc56fe690
md"Which is pretty close to our answer, $\frac{1}{3}$"

# ╔═╡ 332020be-9311-11eb-16c5-03374a5c7112
a2 = rand(1:0.00001:10, 10000)

# ╔═╡ 420afebe-9311-11eb-1cd2-2b0ddbb58716
b2 = a2[a2 .<= 3]

# ╔═╡ 5e578c2c-9311-11eb-22a0-2f57b00d7958
"$(length(b2)/length(a2)) which is close to $(2/9)"

# ╔═╡ 55eee81a-f8cc-4503-adce-3153342baf00
md"### Let's set up some terms"

# ╔═╡ 6f8a52ba-f673-47e0-890a-83a7b2e338d9
md"""
(Ⅰ) **Probability Distribution**

_"In probability theory and statistics, a probability distribution is the mathematical function that gives the probabilities of occurrence of different possible outcomes for an experiment."_

$ \,\,\,\,\,\,\,\,\,\,\,\,\,\,\, $ --[Wikipedia](https://en.wikipedia.org/wiki/Probability_distribution)
"""

# ╔═╡ 48bffce6-6ba1-44b8-bd70-ff11b708a23d
md"""
(Ⅱ) **Mean**

Its the average of a probability distribution of a discrete random variable.
"""

# ╔═╡ b5b45043-cbea-4848-862c-5b39bf6fd2bb
md"""
(Ⅲ) **Standard Deviation**

_"In statistics, the standard deviation is a measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean of the set, while a high standard deviation indicates that the values are spread out over a wider range."_

$ \,\,\,\,\,\,\,\,\,\,\,\,\,\,\, $ --[Wikipedia](https://en.wikipedia.org/wiki/Standard_deviation)

"""

# ╔═╡ a75111b9-063a-4158-a78f-7f1107db028c
md"""
(Ⅳ) **Expectation**

The expectation $E$ is defined for a discrete random variable as 

$E = ∑P(x)f(x)$
where it tells the expected value of some function $f(x)$, if its probability distribution is $P(x)$.

Similarly, the expectation E is defined for a discrete random variable as 

$E = ∫p(x)f(x)dx$
where it tells the expected value of some function $f(x)$, if its probability distribution is $p(x)$.
"""

# ╔═╡ 1683d427-5433-4d85-b65e-4b86cc61af14
md"""
(Ⅴ) **Variance**

**Definition from the book:** _"The variance gives a measure of how much the values of a function of a random
variable x vary as we sample diﬀerent values of x from its probability distribution."_

$Var(f(x)) = E[(f(x) − E[f(x)])^2]$
**_Note_**:
$Var(f(x)) = (Standard\,Deviation)^2$
"""

# ╔═╡ 3a2c5692-ee39-4f26-b094-a6094dd88b68
md"""
### Some commonly used Probability Distributions
"""

# ╔═╡ f5af9c66-ee47-4384-a68a-60711f966557
md"#### [Bernoulli Distribution](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Bernoulli)"

# ╔═╡ 203d14b2-d7b6-4ac4-b1f8-421a2e80c51a
bern = Bernoulli()

# ╔═╡ e18cdf58-13a9-4f98-ab97-6c5cdcc3ee7c
mean(bern), var(bern) #mean and variance of bernoulli distribution

# ╔═╡ a0dfaf17-1cbf-45e1-be14-8302f68b2b6a
md"#### [Multinomial Distribution](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.Multinomial)"

# ╔═╡ 052c45aa-1735-4cba-b911-28c81d7e2850
let
	m = Multinomial(5, 12) #setting up the mulinomial distribution, with n=5
	m, mean(m)
end

# ╔═╡ f2a98e2b-328c-4a8d-bfeb-e5dd7cbb3014
let
	m = Multinomial(12, [0.3, 0.2, 0.45, 0.05])
	m, var(m)
end

# ╔═╡ 6f8963c0-8c36-43ab-9a6d-e814aaba2d81
md"#### Multinoulli Distribution"

# ╔═╡ 4ee83e69-db95-4724-b963-de7dd2445b02
multinoulli = Multinomial(1, 6) #setting up the mulinoulli distribution with k=6

# ╔═╡ 1de8aded-81e9-4f87-baa2-c571f31cb31b
md"""
#### Gaussian Distribution
"""

# ╔═╡ d038c84a-af8b-4da7-974d-0c023e40fcfb
md"[Univariate Normal Distribution](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Normal)"

# ╔═╡ d5b23092-a9fe-44c3-ae9d-bd7186cbdf07
uninorma = Normal(1, 0.2)

# ╔═╡ f7f637b6-2f6e-47dc-b1ca-baf1669f6681
mean(uninorma), var(uninorma)

# ╔═╡ 7cff52e3-5063-47a6-8d0e-03c333ee401b
md"[Multivariate Normal Distribution](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.AbstractMvNormal)"

# ╔═╡ 0235d738-c7e4-48b0-b7fd-9b6d31a2bbbc
multinorma = MvNormal(3, 0.8)

# ╔═╡ 0b67edef-d59a-44ed-9226-3c857e78b0e7
mean(multinorma), var(multinorma)

# ╔═╡ d3faf6d4-6584-476f-bb62-d34ccb701bc4
rand(bern), rand(multinorma), rand(uninorma), rand(multinoulli), rand(Multinomial(5, 12))

# ╔═╡ ce6a1226-923c-11eb-074b-6d708b9773ff
md"""
!!! question "Question"
	Heights of  $25$-year-old men in a certain region have mean $69.75$ inches and standard deviation  $2.59$  inches. These heights are approximately normally distributed. Thus the height  X  of a randomly selected  $25$-year-old man is a normal random variable with mean $μ=69.75$ and standard deviation  $σ=2.59$. Find the probability that a randomly selected $25$-year-old man is more than $69.75$ inches tall.
"""

# ╔═╡ 6489f711-b127-4283-8171-9478d4c48a0c
n = Normal(69.75, 2.59)

# ╔═╡ fc4769e0-f6c1-473c-b675-13ec737a5572
inputrange = quantile(n, 0.0001):0.1:quantile(n, 1-0.0001) #outputs a set of values between the minimum and maximum of an infinite distribution, with an interval of 0.1 in this case

# ╔═╡ 5a8346f0-ea6f-43e6-916b-041ee5220367
 output = pdf(n, inputrange)

# ╔═╡ e1be7ec4-6121-401b-b338-37cd34591586
let
	plot(inputrange, output, Axes(xtics=60.75:3:80, title="'Density function of the height of a 25 year old man'"), w=:l)
	d = findfirst(output .== maximum(output))
	plot!(repeat([inputrange[d]],2), [0, output[d]], w=:lp, pt=7, lt=:dash)
end

# ╔═╡ 251004a8-fb13-4e92-8c08-27286351753f
md"""
As you can see, the plot is symmetric with respect to $x=69.75$. It means, the probability of a randomly selected $25$ year old person being greater than $69.75$ inches in height is same as the probability of that person's height being less than $69.75$ inches.

Hence, outcomes are equally likely and we can say that

$p(x>69.75) = p(x<69.75) = 0.5$
"""

# ╔═╡ 291d53fc-93ec-4926-80ab-a276b025ef82
md"""
## Likelihood
"""

# ╔═╡ fd6e90a0-36f9-4055-a69b-5809cd05b837


# ╔═╡ 3d141deb-bf7c-4ef3-88db-1b7e31ba6033
md"""
#### Try Yourself
1. Try making the Dirac Delta function in Julia.
2. Explore [this](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/05%3A_Continuous_Random_Variables/5.01%3A_Continuous_Random_Variables) website, look at solved and unsolved questions, try solving them in julia.
3. Check out the [Exponential Distribution](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Exponential) and [Laplace Distribution](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Laplace) from Distributions.jl
4. Explore [Gaston](https://mbaz.github.io/Gaston.jl/stable/) more, as it'll be used throughout the tutorial. Try getting familiar with it.
"""

# ╔═╡ d4bf1c39-6281-4ddf-b87e-bf4e3cefea3e
md"# Information Theory"

# ╔═╡ 747b4b5b-98e7-4171-bfb0-b5acc792eda3
md"""
The self-information of an event $X=x$ denoted by 𝐼, is given by
"""

# ╔═╡ 75b07946-f493-49f0-928c-afee87fb22ad
I(x) = -log(x)

# ╔═╡ 4cf8dd4f-af3b-421f-ab11-e3036d60cc29
md"where x denotes the probability of a random variable's outcome being $x$."

# ╔═╡ 2454387d-df18-4932-ad48-691e5583a21e
md"with the Shanon Entropy given by"

# ╔═╡ 50920106-7a39-4563-a1c0-cdd5d43ebb37
H(x) = mean(I(x))

# ╔═╡ e8140a83-904a-46b6-a8e8-8a7151b85c43
md"Its mostly theoritical stuff. In this section, I'd like to introduce the reader to Graphs"

# ╔═╡ 76dc38a7-d06e-41c4-8da5-b15e3ebe54de
md"""
## Graphs
_In mathematics, and more specifically in graph theory, a graph is a structure amounting to a set of objects in which some pairs of the objects are in some sense "related". The objects correspond to mathematical abstractions called **vertices** (also called **nodes** or **points**) and each of the related pairs of vertices is called an **edge** (also called link or line). Typically, a graph is depicted in diagrammatic form as a set of dots or circles for the vertices, joined by lines or curves for the edges._

_The edges may be **directed** or **undirected**. For example, if the vertices represent people at a party, and there is an edge between two people if they shake hands, then this graph is undirected because any person A can shake hands with a person B only if B also shakes hands with A. In contrast, if any edge from a person A to a person B corresponds to A owes money to B, then this graph is **directed**, because owing money is not necessarily reciprocated. The former type of graph is called an **undirected** graph while the latter type of graph is called a **directed** graph._

$ \,\,\,\,\,\,\,\,\, $ --[Wikipedia](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics))
"""

# ╔═╡ 79f9cf36-1f2c-456f-a71c-d0bac25a5aad
md"For an example, lets take a look ath the below graph."

# ╔═╡ 87f96fa0-c27e-45a6-8452-f67f952b737d
Resource("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Undirected.svg/220px-Undirected.svg.png")

# ╔═╡ 1d009804-1858-4109-a75b-16ae5523bfa2
md"The three blue circles are **nodes**. While the lines connecting them are **edges**"

# ╔═╡ cdf1583c-6964-4064-9434-8be1d4c397d9
md"Its an **undirected graph**, since the edges have no orientation."

# ╔═╡ 68d60332-1aa2-4668-bfbe-fefa588fe582
md"Lets try making one in LightGraphs"

# ╔═╡ b7c3798f-5fbb-4a87-8299-28e4c4c28fec
let
	g = SimpleGraph(3) #Makes an undirected graph
	add_edge!(g, 1, 2); #Adding an edge from 1 to 2
	add_edge!(g, 2, 3);
	add_edge!(g, 1, 3);
	gplot(g)
end

# ╔═╡ cf8483c1-d805-4e99-b2f4-7dce289fa587
q = [[1,1,1], [1,1,1]]

# ╔═╡ e07b27ab-b1c5-41c6-b932-641bcd893324
append!(q[1], 1)

# ╔═╡ 236e38f8-acea-4d03-a34d-d502e6e8e32d
md"""
A direct graph is one where the edges have an orientation. From the above example, assume A owes money to B and B owes money to C.
"""

# ╔═╡ a686295f-6cf1-4da4-be0e-b1935892fc44
let
	g = SimpleDiGraph(3) #Makes an undirected graph
	add_edge!(g, 1, 2); #Adding an edge from 1 to 2
	add_edge!(g, 2, 3);
	gplot(g, nodelabel=["A", "B", "C"]) 
end

# ╔═╡ 26506fba-3d46-49e5-822b-725741003e17
md"
A **Weighted Graph** is a graph where weights are assigned to each edge.
Assume, in the above example, A owes `` \$54 `` to B and B owes `` \$89 `` to C. 
"

# ╔═╡ df7e83b8-eb8b-411c-be35-07d5770fe9b1
let
	sources = [1,2]
	destinations = [2,3]
	weights = [54, 89]
	nodelabel=["A", "B", "C"]
	g = SimpleWeightedDiGraph(sources, destinations, weights) #Makes an undirected graph
	gplot(g, nodelabel=["A", "B", "C"], edgelabel=["５𝟦", "𝟾𝟫"])
end

# ╔═╡ 0119dc9f-8ccf-4776-a7e9-5f61b063d509
md"
### Try yourself
"

# ╔═╡ 0db8895b-cb4d-4a3d-9bae-1fd0f460fafe
md"""
1. Look for different types of graphs on the [wikipedia page](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)). Try making and visualizing them using [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl)+[GraphPlot](https://github.com/JuliaGraphs/GraphPlot.jl).
2. Try visualizing both the structured probablistic models examples.
"""

# ╔═╡ Cell order:
# ╟─ea0dc819-e860-4863-b9ea-81d12d92c4ab
# ╟─e84a31da-8ce7-11eb-1b6a-0f1431d8c96e
# ╟─48727d46-8cf8-11eb-25a0-73c248dab2da
# ╟─af0e8170-9163-11eb-2e48-4dcc7c853090
# ╟─dde86ab8-9163-11eb-173a-0d0fe5d297ad
# ╟─51cf0da0-8cf8-11eb-13d1-53bb6bd3892a
# ╟─5c51aa26-8cf8-11eb-0add-a1b7f41216f6
# ╟─72edf972-8cf8-11eb-2441-75b6edc68854
# ╠═c26eb560-9167-11eb-02d3-8b4212f378e6
# ╟─6d6ead0a-9170-11eb-049b-2dfaf08ba40e
# ╟─5e59cfb6-9170-11eb-0215-2390218aac63
# ╠═6fd98eb6-916b-11eb-19a4-5368d9cc2b31
# ╟─980bb77e-9170-11eb-3d61-19044344aa37
# ╠═0c0ac120-916b-11eb-1029-cb8615bca99c
# ╟─4ef0b236-9172-11eb-0813-e12589886a31
# ╠═47f31ab4-9172-11eb-1bc4-f916418a47db
# ╟─eced0ac6-9171-11eb-0f2a-cd5fa1bd818d
# ╟─3f168c24-9173-11eb-1cf8-7338a422ff69
# ╟─f6b2bf60-9173-11eb-145d-2f1f47b7eb30
# ╟─79557c7c-9173-11eb-3c53-4fa9ad783e8a
# ╟─35cfa732-9238-11eb-3f47-0d921b5d6eeb
# ╟─b4374be0-9173-11eb-09d2-cbab6657cdaf
# ╟─70e11290-923b-11eb-2365-f58a04d0f844
# ╟─758a1460-8cf8-11eb-0ab7-7dfcfac203f9
# ╟─806857e8-8cf8-11eb-3756-bff672168730
# ╟─7bc55826-92de-11eb-258b-31d565464c56
# ╟─a2128026-92de-11eb-102f-d924f0ceee27
# ╟─749c569c-92e0-11eb-14c5-8301f513cdad
# ╟─e0c4991e-92e1-11eb-1cc3-31c0fae42abf
# ╠═1c95027c-930e-11eb-2a05-8322da402360
# ╠═a6073126-930c-11eb-2785-79a71976e4d5
# ╠═f9c89c3c-930c-11eb-278d-9bf0ee036c67
# ╟─16b2a794-9311-11eb-3604-f1efc56fe690
# ╠═332020be-9311-11eb-16c5-03374a5c7112
# ╠═420afebe-9311-11eb-1cd2-2b0ddbb58716
# ╠═5e578c2c-9311-11eb-22a0-2f57b00d7958
# ╟─55eee81a-f8cc-4503-adce-3153342baf00
# ╟─6f8a52ba-f673-47e0-890a-83a7b2e338d9
# ╟─48bffce6-6ba1-44b8-bd70-ff11b708a23d
# ╟─b5b45043-cbea-4848-862c-5b39bf6fd2bb
# ╟─a75111b9-063a-4158-a78f-7f1107db028c
# ╟─1683d427-5433-4d85-b65e-4b86cc61af14
# ╟─3a2c5692-ee39-4f26-b094-a6094dd88b68
# ╠═02d67371-b1e3-4ac9-b4d1-cdca1c85bd18
# ╟─f5af9c66-ee47-4384-a68a-60711f966557
# ╠═203d14b2-d7b6-4ac4-b1f8-421a2e80c51a
# ╠═e18cdf58-13a9-4f98-ab97-6c5cdcc3ee7c
# ╟─a0dfaf17-1cbf-45e1-be14-8302f68b2b6a
# ╠═052c45aa-1735-4cba-b911-28c81d7e2850
# ╠═f2a98e2b-328c-4a8d-bfeb-e5dd7cbb3014
# ╟─6f8963c0-8c36-43ab-9a6d-e814aaba2d81
# ╠═4ee83e69-db95-4724-b963-de7dd2445b02
# ╟─1de8aded-81e9-4f87-baa2-c571f31cb31b
# ╟─d038c84a-af8b-4da7-974d-0c023e40fcfb
# ╠═d5b23092-a9fe-44c3-ae9d-bd7186cbdf07
# ╠═f7f637b6-2f6e-47dc-b1ca-baf1669f6681
# ╟─7cff52e3-5063-47a6-8d0e-03c333ee401b
# ╠═0235d738-c7e4-48b0-b7fd-9b6d31a2bbbc
# ╠═0b67edef-d59a-44ed-9226-3c857e78b0e7
# ╠═d3faf6d4-6584-476f-bb62-d34ccb701bc4
# ╟─ce6a1226-923c-11eb-074b-6d708b9773ff
# ╠═7c8394db-271e-4d0b-a3db-b3e3a659ec0a
# ╠═6489f711-b127-4283-8171-9478d4c48a0c
# ╠═fc4769e0-f6c1-473c-b675-13ec737a5572
# ╠═5a8346f0-ea6f-43e6-916b-041ee5220367
# ╠═e1be7ec4-6121-401b-b338-37cd34591586
# ╟─251004a8-fb13-4e92-8c08-27286351753f
# ╟─291d53fc-93ec-4926-80ab-a276b025ef82
# ╠═fd6e90a0-36f9-4055-a69b-5809cd05b837
# ╟─3d141deb-bf7c-4ef3-88db-1b7e31ba6033
# ╟─d4bf1c39-6281-4ddf-b87e-bf4e3cefea3e
# ╟─747b4b5b-98e7-4171-bfb0-b5acc792eda3
# ╠═75b07946-f493-49f0-928c-afee87fb22ad
# ╟─4cf8dd4f-af3b-421f-ab11-e3036d60cc29
# ╟─2454387d-df18-4932-ad48-691e5583a21e
# ╠═50920106-7a39-4563-a1c0-cdd5d43ebb37
# ╟─e8140a83-904a-46b6-a8e8-8a7151b85c43
# ╠═bf7ce886-e6ac-45e7-a0f5-98e36c8974c7
# ╟─76dc38a7-d06e-41c4-8da5-b15e3ebe54de
# ╠═7a514f20-e7cc-43da-96f6-cc4f7244166b
# ╟─79f9cf36-1f2c-456f-a71c-d0bac25a5aad
# ╠═87f96fa0-c27e-45a6-8452-f67f952b737d
# ╟─1d009804-1858-4109-a75b-16ae5523bfa2
# ╟─cdf1583c-6964-4064-9434-8be1d4c397d9
# ╟─68d60332-1aa2-4668-bfbe-fefa588fe582
# ╠═b7c3798f-5fbb-4a87-8299-28e4c4c28fec
# ╠═cf8483c1-d805-4e99-b2f4-7dce289fa587
# ╠═e07b27ab-b1c5-41c6-b932-641bcd893324
# ╟─236e38f8-acea-4d03-a34d-d502e6e8e32d
# ╠═a686295f-6cf1-4da4-be0e-b1935892fc44
# ╟─26506fba-3d46-49e5-822b-725741003e17
# ╠═c7a5ab0b-aba0-4b54-abe9-702b0bad4a8e
# ╠═df7e83b8-eb8b-411c-be35-07d5770fe9b1
# ╟─0119dc9f-8ccf-4776-a7e9-5f61b063d509
# ╟─0db8895b-cb4d-4a3d-9bae-1fd0f460fafe
