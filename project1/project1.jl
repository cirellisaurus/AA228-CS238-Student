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

    index_array = collect(1:length(vars.names))


    network_DiGraph, score = Perform_K2_Search(index_array, vars, D_matrix)
    order_best = index_array
    score_best = score
    graph_best = network_DiGraph
    println("baseline order: $index_array")
    println("baseline bayesian score: $score_best")
    for attempt in 1:3
        #create ordering array for K2Search 
        ordering_array = shuffle(index_array)
        #find optimal structure 
        network_DiGraph, score = Perform_K2_Search(ordering_array, vars, D_matrix)
        println("bayesian score: $score")
        if score > score_best
            # update the current score with the best score and update best order 
            score_best = score
            order_best = ordering_array
            graph_best = network_DiGraph

        end
    end 
    println("best order: $order_best")
    println("best bayesian score: $score_best")



    
    #graph for testing: 
    #  network_DiGraph = DiGraph(6)
    #  add_edge!(network_DiGraph, 1, 2)
    #  add_edge!(network_DiGraph, 3, 4)
    #  add_edge!(network_DiGraph, 5, 6)
   
    # compute M testing 
    # M = computeM(vars, network_DiGraph, D_matrix)

    #report bayesian score for optimal structure
    #b = bayesian_score(vars, network_DiGraph, D_matrix)

    #println("bayesian score: $b")

    #create a Dict for write_gph that matches the var names to their indices
    idx2names_Dict = Dict(zip(index_array, vars.names))

    #write gph file 
    write_gph(graph_best, idx2names_Dict, outfile_gph)


    #create graph visualization TikZGraphs
    # documentation: https://github.com/JuliaTeX/TikzGraphs.jl/blob/master/doc/TikzGraphs.ipynb
    t = TikzGraphs.plot(graph_best)

    #save graph to pdf 
    TikzPictures.save(PDF(infile), t)

    #other options for saving the graph 
    #TikzPictures.save(SVG("graph"), t)
    #TikzPictures.save(TEX("graph"), t)

end

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

#This function computes the count matrix M 
function computeM(r, G, D::Matrix{Int}, M, n)
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
                #println("dims: $dims")
                #println("dims type: $(typeof(dims))")
                #println("dims size: $(size(dims))")

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

#this function calculates the Bayesian score summation 
function calculate_p(M, prior, n)
    total = 0
    for i in 1:n
        total += sum(loggamma.(prior[i] + M[i])) - sum(loggamma.(prior[i])) + sum(loggamma.(sum(prior[i],dims=2))) - sum(loggamma.(sum(prior[i],dims=2) + sum(M[i],dims=2)))
    end
    return total
end


function bayesian_score(vars, G, D)
    # set n equal to number of columns in data
    n = size(D, 2)
    #put counts for vars into array
    r = vars.r
    #println("r: $r")
    #println("r type: $(typeof(r))")

    # calculate q by calculating the product of the number of parential instantiations for node i for each node 
    q = Int.(ones(n))
    #println("q type: $(typeof(q))")

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
    M = computeM(r, G, D, M, n)

    prior = [ones(q[i], r[i]) for i in 1:n]

    p = calculate_p(M, prior, n)
    return p 
end

# method is an argument of type K2Search, which means that the input to this argument must be an instance of the K2Search struct.
# vars is an argument of type Variable, which means that the input to this argument must be an instance of the Variable struct.
# D is an argument of type Matrix{Int}, which means that the input to this argument must be a two-dimensional array of integers.
# D, the data set is an m x n matrix, where n is the number of variables and m is the number of data points
function Perform_K2_Search(ordering, vars, D)
    #create a directed graph (SimpleDiGraph) with length(vars) number of nodes
    G = SimpleDiGraph(length(vars.names))
    #ind = index, child = element
    #for each child starting with the 2nd variable in the ordering
    score_curr = -Inf
    for (ind,child) in enumerate(ordering[2:end])

        #calculate bayesian score of current graph
        score_curr = bayesian_score(vars, G, D)
        #print("Graph: $G \n")
        #print("Current Bayesian score of graph: $y \n")

        #while score can still be improved 
        while true
            # we set our baseline Bayesian score to -infinity and try to improve from there 
            score_best = -Inf
            # set best parent to placeholder value of 0
            parent_best = 0           
            #for parents at index 1 through index ind in the ordering array 
            for parent in ordering[1:ind]
                
                #if there is no edge between the current parent and current child
                #has_edge(G, parent, child) is a function that checks if there is an edge in a graph G between the vertices parent and child
                if !has_edge(G, parent, child)
                    # Add edge to our graph 
                    add_edge!(G, parent, child)
                    #calculate what the new score is with the new edge added 
                    score_new = bayesian_score(vars, G, D)
                    #print("index: $ind, curr child: $child, potential parent, $parent, current Bayes score, $score\n")

                    # if the new score is better than our current best score
                    if score_new > score_best
                        #update the best score and update who the best parent is for that child
                        score_best = score_new
                        parent_best = parent
                    end
                    #Remove edge to our graph - we don't want to actually add the edge until we are sure the edge we are adding is the best edge
                    rem_edge!(G, parent, child)
                end
            end
            #if the score is better when we connect child with best parent
            if score_best > score_curr
                # update the current score with the best score and add the edge connecting the best parent to the child 
                score_curr = score_best
                #print("best parent, $parent_best, current child, $child, best Bayes score, $y_best. \n")
                #add an edge between nodes parent_best and child in a graph G
                add_edge!(G, parent_best, child)
                
            #if score is not better, move onto next child
            else
                break
            end
        end
    end
    return G, score_curr
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

