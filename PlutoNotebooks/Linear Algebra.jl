### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 7f621598-8ce7-11eb-1d78-89ab395c5aac
using MultivariateStats

# ╔═╡ 1e960125-0783-4932-9b98-7b09c72bc951
md"[Reference Chapter](https://www.deeplearningbook.org/contents/linear_algebra.html)"

# ╔═╡ 811b4842-775b-11eb-3690-830857abe1b5
md"### Eigenvectors"

# ╔═╡ ec3044cc-7768-11eb-17ff-c19d9ac19ae4
md"
Cosider a matrix of order ``n × n`` 

$A = \begin{bmatrix}a_{11} & a_{12} & a_{13} & a_{14} & ... & a_{1n}\\a_{21} & a_{22} & a_{23} & a_{24} & ... & a_{2n}\\a_{31} & a_{32} & a_{33} & a_{34} & ... & a_{3n}\\a_{41} & a_{42} & a_{43} & a_{44} & ... & a_{4n}\\\vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\a_{n1} & a_{n2} & a_{n3} & a_{n4} & ... & a_{nn}\end{bmatrix}$

When we multiply it by a vector of length `` n ``...

$B = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \\\vdots \\ b_n \end{bmatrix}$

`` C = AB ``

Therefore, ``C = \begin{bmatrix}a_{11} & a_{12} & a_{13} & a_{14} & ... & a_{1n}\\a_{21} & a_{22} & a_{23} & a_{24} & ... & a_{2n}\\a_{31} & a_{32} & a_{33} & a_{34} & ... & a_{3n}\\a_{41} & a_{42} & a_{43} & a_{44} & ... & a_{4n}\\\vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\a_{n1} & a_{n2} & a_{n3} & a_{n4} & ... & a_{nn}\end{bmatrix}$ $\begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \\\vdots \\ b_n \end{bmatrix}`` 

which can be rewritten as,

$C = \begin{bmatrix} a_{11} \times b_1 + a_{12} \times b_2 + a_{13} \times b_3 + a_{14} \times b_4 ... a_{1n} \times b_n \\ a_{21} \times b_1 + a_{22} \times b_2 + a_{23} \times b_3 + a_{24} \times b_4 ... a_{2n} \times b_n \\ a_{31} \times b_1 + a_{32} \times b_2 + a_{33} \times b_3 + a_{34} \times b_4 ... a_{3n} \times b_n \\ a_{41} \times b_1 + a_{42} \times b_2 + a_{43} \times b_3 + a_{44} \times b_4 ... a_{4n} \times b_n \\\vdots \\ a_{n1} \times b_1 + a_{n2} \times b_2 + a_{n3} \times b_3 + a_{n4} \times b_4 ... a_{nn} \times b_n \end{bmatrix}$

If there exists a scalar `` \lambda ``, such that `` B \times \lambda = C ``, then `` B `` is the eigenvector of `` A ``, and `` \lambda `` is its eigenvalue.
"

# ╔═╡ dd42dfa8-7aa9-11eb-3312-fbed21eb12d5
md"## Eigendecomposition"

# ╔═╡ ec9fae4a-7aa9-11eb-10a9-bfed672dfaca
md"If a matrix 𝐴 has 𝑛 linearly independent eigenvectors and 𝑛 corresponding eigenvalues, then if we make a matrix out of the eigenvectors, and we call it 𝑉, where every column in 𝑉 is an eigenvector of 𝐴, and make a (column)vector 𝜆 which has eigenvalues, corresponding to eigenvectors of 𝐴, then we can say that :

Eigendecomposition of 𝐴 = 𝑉_diag_(𝜆)𝑉$^{-1}$
" 

# ╔═╡ 6dca1270-7aab-11eb-23a6-c5db84f32eb5
let
	import LinearAlgebra
	A = rand(1:10,3,3)
	F = LinearAlgebra.eigen(A)
	A, F
end

# ╔═╡ 5b37c0c6-7ab5-11eb-2fe2-4fefc5a9e193
md"""
In the above code, we find the eigendecomposition of 𝐴, and store it in 𝐹, where `F.vectors` give us all the eigenvalues(or the matrix 𝑉 in the above context), and `F.values` stores the corresponding eigenvalues(the vector 𝜆 in the above context).
"""

# ╔═╡ 8f59c323-4326-4c5a-b013-6e0cfa442e02
md"""
!!! note "Some terms"

	If a matrix's eigenvalues are all positive, its called _**positive definite**_.

	If a matrix's eigenvalues are all positive or zero, its called _**positive semidefinite**_.

	If a matrix's eigenvalues are all negative, its called _**negative definite**_.

	If a matrix's eigenvalues are all negative or zero, its called _**negative semidefinite**_.

	In a **_positive semidefinite_** matrix, _for all x, **xᵀ𝐴x ≥ 𝟢**_.

	In a **_positive semidefinite_** matrix, _**xᵀ𝐴x = 𝟢**, implies that,_ **_x = 𝟢_**.
"""

# ╔═╡ 11a7a0f0-7ad8-11eb-12c7-77f971d32a1a
md"## Singular Value Decomposition"

# ╔═╡ 2c663d90-7ad8-11eb-0e6c-039ecc187c0d
md"""
There's another method of decomposing (or factorising) a matrix into sigular vectors and singular values called Singular Value Decomposition.
Every real matrix has a singular value decomposition, but not every matrix has can be decomposed into eigenvalues. For example, matrices which are not square.

#### What is it?
We can write a matrix 𝐴 as a product of 𝟥 matrices --

`` 𝐴 = 𝑈𝐷𝑉^T ``

1. 𝐴 and 𝐷, both have dimensions `` m × n ``, 𝑈 has dimensions `` m × m ``, and 𝑉 has dimensions `` n × n ``.
2. 𝑈 and 𝑉 are both orthogonal matrices.
3. 𝐷 is a diagonal matrix and is not necessarily square.
4. The diagonal elements of 𝐷, also known as singular values of matrix 𝐴, are the are the square roots of eigenvalues of 𝐴ᵀ𝐴 or 𝐴𝐴ᵀ(eigenvalues of both are equal).
5. The columns of 𝑈, also known as left-singular vectors of 𝐴, are the eigenvectors of 𝐴𝐴ᵀ.
6. The columns of 𝑉, also known as right-singular vectors of 𝐴, are the eigenvectors of 𝐴ᵀ𝐴.
"""

# ╔═╡ bdef0196-7b91-11eb-2d26-7b02fa67233c
let
	A = convert(Matrix,rand(0:9, 3,3))
	LinearAlgebra.svd(A)
end

# ╔═╡ d0fd5526-7d00-11eb-113b-47a224413bac
md"## Trace operator
A simple demonstration would be easier to understand."

# ╔═╡ 22247216-7ddf-11eb-24da-65455796685f
let
	A = convert(Matrix,rand(0:9, 3,3))
	A, LinearAlgebra.tr(A)
end

# ╔═╡ 052770a6-787d-11eb-1816-11fc224a68bb
md"It gives the sum of diagonal elements."

# ╔═╡ 0085a0f2-7ded-11eb-18c3-8b40f3a71ab9
md"## Moore Penrose Pseudo-Inverse
Assume there exists a matrix 𝐴, and its left-inverse 𝐵. Such that they satisfy,

`` 𝐴𝑥 = 𝑦 ``

`` 𝑥 = 𝐵𝑦  ``

The pseudo-inverse of 𝐴 is $A^+$, where $A^+$ given by the matrix,

`` 𝐴^+ = VD^+U^T ``

𝑈, 𝑉 and 𝐷 are from the Singular Value Decomposition of 𝐴, as defined above

`` D^+ `` is given by calculating the reciprocal of every non-zero element in 𝐷, and then the resulting matrix's transpose.

Lets just say that `` A^+ `` is pretty close to 𝐵.
"

# ╔═╡ 98be78fe-7dfd-11eb-0812-59df1018f0af
let
	A = convert(Matrix,rand(0:9, 3,3))
	A, LinearAlgebra.pinv(A), inv(transpose(A) * A) * transpose(A)#original matrix, pseudo-inverse and left-inverse
end

# ╔═╡ 0def7a46-787d-11eb-189c-8151f676c8f1
md"## Try It Yourself"

# ╔═╡ dce0bdfb-23c3-415a-9b5c-833782e223a8
md"
**Try understanding the _Principal Component Analysis_ from the book and implementing it in Julia.**
"

# ╔═╡ 22557a0e-7de0-11eb-2fa0-07c807d98310
md"""
Lets understand some terms!

##### [Features](https://www.datarobot.com/wiki/feature/)
Assume you've to predict the price of houses. The data you have is, size of houses, size of lawn, number of rooms, and of course, price.
Here, size of house, size of lawn, number of rooms, and price act as features. Also note that here, price is the _dependent variable_, while rest are independent variables.

##### Feature Elimination
It can be hard to work with multiple features. From the above features, you can try eliminating one or more features, which will better help in prediction. The accuracy of prediction will fall though. For an example, I can assume that the size of lawn variable doesn't exist here. Working with the two remaining independent variables to predict the dependent variable (_price_ in the above case), would be easier, but accuracy of prediction will fall.

##### Feature Extraction
Dropping a whole feature can't be very reliable, you say? Yeah it isn't (in many cases)! Hence, introducing _feature extraction_! We can try making new independent features from the ones we already have, but has overall lesser dimensions in a way that we can make the data in the new features, as accurate as possible compared to the old ones. Since every new feature is made from the combining old ones, all new features have much of the information of old ones!

So, is there some specified way to make those new features I talk about, in feature extraction, you ask? Yes! There are many, and one such way is called **_Principle Component Analysis_**.

Principle Component Analyis is a method of doing feature extraction.

To start, the reference book and [this article](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c) are good to read.
"""

# ╔═╡ 73ef6fad-08fc-4294-8306-68ed5722a8bb
md"
_Hint_: [MultivariateStats.jl](https://multivariatestatsjl.readthedocs.io/en/latest/pca.html)
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"

[compat]
MultivariateStats = "~0.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0-beta3.0"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[deps.Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─1e960125-0783-4932-9b98-7b09c72bc951
# ╟─811b4842-775b-11eb-3690-830857abe1b5
# ╟─ec3044cc-7768-11eb-17ff-c19d9ac19ae4
# ╟─dd42dfa8-7aa9-11eb-3312-fbed21eb12d5
# ╟─ec9fae4a-7aa9-11eb-10a9-bfed672dfaca
# ╠═6dca1270-7aab-11eb-23a6-c5db84f32eb5
# ╟─5b37c0c6-7ab5-11eb-2fe2-4fefc5a9e193
# ╟─8f59c323-4326-4c5a-b013-6e0cfa442e02
# ╟─11a7a0f0-7ad8-11eb-12c7-77f971d32a1a
# ╟─2c663d90-7ad8-11eb-0e6c-039ecc187c0d
# ╠═bdef0196-7b91-11eb-2d26-7b02fa67233c
# ╟─d0fd5526-7d00-11eb-113b-47a224413bac
# ╠═22247216-7ddf-11eb-24da-65455796685f
# ╟─052770a6-787d-11eb-1816-11fc224a68bb
# ╟─0085a0f2-7ded-11eb-18c3-8b40f3a71ab9
# ╠═98be78fe-7dfd-11eb-0812-59df1018f0af
# ╟─0def7a46-787d-11eb-189c-8151f676c8f1
# ╟─dce0bdfb-23c3-415a-9b5c-833782e223a8
# ╟─22557a0e-7de0-11eb-2fa0-07c807d98310
# ╟─73ef6fad-08fc-4294-8306-68ed5722a8bb
# ╠═7f621598-8ce7-11eb-1d78-89ab395c5aac
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
