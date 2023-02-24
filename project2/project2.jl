using Printf
using CSV
using DataFrames
using SparseArrays
using Statistics

"""
    write_policy(, idx2names, filename)
    Each output file should contain an action for every possible state in the problem. 
    The i-th row contains the action to be taken from the i-th state. 
    small.policy should contain 100 lines, medium.policy should contain 50000 lines, and large.policy should contain 312020 lines. 
    Each line should have a number between 1 and the total number of actions for that problem. 
    If using Linux/MAC, you can check the number of lines using wc -l <file> e.g. wc -l small.policy

"""

function write_policy(Q, filename)
    open(filename, "w") do io
        # for edge in edges(dag)
        for Q_row in eachrow(Q)
            @printf(io, "%s \n", argmax(Q_row))
        end
    end
end


function compute(infile, outfile_policy)

    #read CSV file into dataframe 
    df = CSV.read(infile, DataFrame)

    alpha = .9 #learning parameter
 
    epsilon = .01 #minimum rmse before consider Q matrix to have converged


    if infile == "data/small.csv"
        
        gamma = .95 #discount factor
        num_states = 100
        num_actions = 4
    elseif infile == "data/medium.csv"
        gamma = 1 #discount factor
        num_states = 50000
        num_actions = 7
    elseif infile == "data/large.csv"
        gamma = .95 #discount factor
        num_states = 312020
        num_actions = 9 
    end
        
        
    s = 1
    a = 2
    r = 3
    sp = 4

    Q = zeros(num_states, num_actions)
    Q_old = zeros(num_states, num_actions)

    num_updates = 0
    while true
        # for each row in dataset (indexing by i) 
        for df_row in eachrow(df)    
            Q[df_row.s, df_row.a] = Q[df_row.s,df_row.a] + alpha*(df_row.r+gamma*maximum(Q[df_row.sp,:]))
        end 
        rmse = sqrt(mean((Q - Q_old).^2))
        println("rmse $rmse")
        #if Q is signficantly different from Q_old, update Q_old to be Q and reupdate Q again
        if rmse > epsilon
            Q_old = Q
            num_updates += 1
            
        #if Q is converging, exit the loop 
        else
            break
        end   
    end 

    println("num_updates $num_updates")


    # small_num_states = 100
    # small_num_actions = 4

    # T_action1 = spzeros(small_num_states, small_num_states)
    # T_action2 = spzeros(small_num_states, small_num_states)
    # T_action3 = spzeros(small_num_states, small_num_states)
    # T_action4 = spzeros(small_num_states, small_num_states)


    # for i in small_num_actions
    #     subset_df = df[df.a .== i, :]
    # end

    #write policy file 
    write_policy(Q, outfile_policy)


end




# inputfilename = "data/small.csv"
# outputfilename = "output/small.policy"

# inputfilename = "data/medium.csv"
# outputfilename = "output/medium.policy"

inputfilename = "data/large.csv"
outputfilename = "output/large.policy"


@time compute(inputfilename, outputfilename)
