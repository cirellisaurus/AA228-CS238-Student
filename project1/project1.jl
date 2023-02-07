using Graphs
using Printf
using CSV
using DataFrames
using TikzGraphs
using TikzPictures
using SpecialFunctions
using Random
using StatsBase
using Plots
using GraphPlot
using BenchmarkTools
using LinearAlgebra


"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end


function compute(infile, outfile_gph)

    #read CSV file into dataframe 
    df = CSV.read(infile, DataFrame)

    #convert df data into Matrix
    # row corresponds to sample, column corresponds to variable
    D_matrix = Matrix(df)
    #println("D_matrix: $D_matrix")

    #Generate r vector of length n: compute number of unique instantiations of each variable in the df    
    r = zeros(length(names(df)))
    i = 1
    for col in names(df)
        r[i] = findmax(unique(df[!, col]))[1]
        #println("r[i]: $(unique(df[!, col])), i: $i")
        i = i+1
        #println("$col has $unique_count unique values")
    end
    #println(r)



    # put xi, variable names, and r, instantiation counts, into Variable struct
    vars = Variable(names(df), r)
    # println(vars)
    # println(vars.names)
    # println(vars.r)

    #create ordering array for K2Search that is just the order of the variables as they show up in the df
    ordering_array = collect(1:length(vars.names))

    #put ordering array into K2Search struct
    ordering_K2Search = K2Search(ordering_array)

    #find optimal structure 
    network_DiGraph = Perform_K2_Search(ordering_K2Search, vars, D_matrix)

    
    #graph for testing: 
    #  network_DiGraph = DiGraph(6)
    #  add_edge!(network_DiGraph, 1, 2)
    #  add_edge!(network_DiGraph, 3, 4)
    #  add_edge!(network_DiGraph, 5, 6)
   
    # compute M testing 
    # M = computeM(vars, network_DiGraph, D_matrix)

    #report bayesian score for optimal structure
    b = bayesian_score(vars, network_DiGraph, D_matrix)

    println("bayesian score: $b")

    #create a Dict for write_gph that matches the var names to their indices
    idx2names_Dict = Dict(zip(ordering_array, vars.names))

    #write gph file 
    write_gph(network_DiGraph, idx2names_Dict, outfile_gph)


    #create graph visualization TikZGraphs
    # documentation: https://github.com/JuliaTeX/TikzGraphs.jl/blob/master/doc/TikzGraphs.ipynb
    t = TikzGraphs.plot(network_DiGraph)

    #save graph to pdf 
    TikzPictures.save(PDF(infile), t)

    #other options for saving the graph 
    #TikzPictures.save(SVG("graph"), t)
    #TikzPictures.save(TEX("graph"), t)

end

"""
Algorithm 4.1. A function for extracting
the statistics, or counts,
from a discrete data set D, assuming
a Bayesian network with variables
vars and structure G. The
data set is an n × m matrix, where
n is the number of variables and
m is the number of data points.
This function returns an array M of
length n. The ith component consists
of a qi × ri matrix of counts.
The sub2ind(siz, x) function returns
a linear index into an array
with dimensions specified by siz
given coordinates x. It is used to
identify which parental instantiation
is relevant to a particular data
point and variable.
"""

#Struct for storing data variable information 
struct Variable
    names::Vector{String}
    r::Vector{Int} # number of possible values
end

#Note: this fxn is copied from https://htmlview.glitch.me/?https://raw.githubusercontent.com/mossr/PlutoNotebooks/master/html/subscript_and_linear_indexing.html
function sub2ind(siz, x)
	k = vcat(1, cumprod(siz[1:end-1]))
	return dot(k, x .- 1) + 1
end


function computeM(vars, G, D::Matrix{Int})
    # set n equal to number of columns in data
    n = size(D, 2)
    #put counts for vars into array
    r = vars.r
    #println("r: $r")
    #println("r type: $(typeof(r))")

    # calculate q by calculating the product of the number of parential instantiations for node i for each node 
    q = Int.(ones(n))
    #println("q type: $(typeof(q))")

    # q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]

    for i in 1:n
        for j in inneighbors(G,i)
            q[i] = (q[i]*r[j])
        end
    end
    #println("q: $q")
    #println("q type: $(typeof(q))")

    #create M array of n matrices (one for each var), where each matrix has dimensions of q_i x r_i 
    M = [zeros(q[i], r[i]) for i in 1:n]
    #println("M: $M")
    #println("M type: $(typeof(M))")


    #for each sample (row)
    for sample in eachrow(D)
        #for each variable (column)
        for i in 1:n
            #k is the value of that node in that sample (since we index from 1 and all data starts from 1, this is fine)
            k = sample[i]
            #println("k: $k")
            #declare parents, an array of the indices of the parents of xi, found using inneighbors(G,i)
            parents = inneighbors(G,i)
            #println("parents: $parents")

            #if the node has parents
            if !isempty(parents)
                
                # setting dims of array equal to r1xr2...
                dims = r[parents]
                #array = Int.(zeros(prod(dims)))
                #println(array)
                #println("parent_instantiations: $parent_instantiations")
                #println(" type: $(typeof(dims))")
                #println("dims size: $(size(dims))")
                
                #println("dims: $dims")
                #println("dims type: $(typeof(dims))")
                #println("dims size: $(size(dims))")
                #instantiation_array = reshape(array, tuple(dims...))
                #println("instantiation array: $instantiation_array")
                # Create a LinearIndices object of size of dims array
                #lin_indices = LinearIndices(instantiation_array)
                #println("lin_indices: $lin_indices")
                #println("lin_indices type: $(typeof(lin_indices))")
                # Define the multi-dimensional indices
                subs = sample[parents]
                #println("subs: $subs")
                #println("subs type: $(typeof(subs))")
                #println("subs size: $(size(subs))")

                # Calculate the linear index
                # j is equal to sub2ind(dims, subs), where dims = (r1, r2) and subs = (i1, i2))
                j = sub2ind(dims, subs)
                #println("j: $j")
            else
                j = 1
            end
            #print("i: $i ")
            #print("sample: $sample ")
            M[i][j,k] += 1.0
            
        end
    end
    #println("M: $M")
    return M
    
end

"""
A function for generating
    a prior αijk where all entries
    are 1. The array of matrices
    that this function returns takes the
    same form as the statistics generated
    by algorithm 4.1. To determine
    the appropriate dimensions, the
    function takes as input the list of
    variables vars and structure G.
"""
function prior(vars, G)
    n = length(vars.names)
    r = [vars.r[i] for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

"""
Algorithm 5.1. An algorithm
for computing the Bayesian score
for a list of variables vars and
a graph G given data D. This
method uses a uniform prior
αijk = 1 for all i, j, and k
as generated by algorithm 4.2.
The loggamma function is provided
by SpecialFunctions.jl. Chapter
4 introduced the statistics
and prior functions. Note that
log(G(α)/G(α + m)) = log G(α) −
log G(α + m), and that log G(1) =
0.
"""
function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
    end
    function bayesian_score(vars, G, D)
    n = length(vars.names)
    M = computeM(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

"""
Algorithm 5.2. K2 search of the
space of directed acyclic graphs using
a specified variable ordering.
This variable ordering imposes a
topological ordering in the resulting
graph. The fit function takes
an ordered list variables vars and
a data set D. The method starts
with an empty graph and iteratively
adds the next parent that
maximally improves the Bayesian
score.
"""

struct K2Search
    ordering::Vector{Int} # variable ordering
end

# method is an argument of type K2Search, which means that the input to this argument must be an instance of the K2Search struct.
# vars is an argument of type Variable, which means that the input to this argument must be an instance of the Variable struct.
# D is an argument of type Matrix{Int}, which means that the input to this argument must be a two-dimensional array of integers.
# D, the data set is an m x n matrix, where n is the number of variables and m is the number of data points
function Perform_K2_Search(method::K2Search, vars, D)
    #create a directed graph (SimpleDiGraph) with length(vars) number of nodes
    G = SimpleDiGraph(length(vars.names))
    #ind = index, child = element

    
    for (ind,child) in enumerate(method.ordering[2:end])
        #y = rand(-100:-1)
        y = bayesian_score(vars, G, D)
        #print("Graph: $G \n")
        #print("Current Bayesian score of graph: $y \n")
        while true
            y_best, parent_best = -Inf, 0
            #for parents 1 through ind
            for parent in method.ordering[1:ind]
                
                #has_edge(G, parent, child) is a function that checks if there is an edge in a graph G between the vertices parent and child
                if !has_edge(G, parent, child)
                    add_edge!(G, parent, child)
                    #check for cycles here? 
                    y_new = bayesian_score(vars, G, D)
                    #y_new = rand(-100:1)
                    #print("index: $ind, curr child: $child, potential parent, $parent, current Bayes score, $y_new\n")
                    if y_new > y_best
                        y_best, parent_best = y_new, parent
                    end
                    rem_edge!(G, parent, child)
                end
            end
            if y_best > y
                y = y_best
                #print("best parent, $parent_best, current child, $child, best Bayes score, $y_best. \n")
                #add an edge between nodes parent_best and child in a graph G
                add_edge!(G, parent_best, child)
                
            else
                break
            end
        end
    end
    return G
end


# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end


# inputfilename = "example.csv"
# outputfilename = "eg.gph"

# inputfilename = "data/small.csv"
# outputfilename = "data/small.gph"

# inputfilename = "data/medium.csv"
# outputfilename = "data/medium.gph"

inputfilename = "data/large.csv"
outputfilename = "data/large.gph"


@time compute(inputfilename, outputfilename)

