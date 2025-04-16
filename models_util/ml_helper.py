# Helper functions for the ML part of the project
# feature engineering, plotting, cost functions and classification metrics,...etc 


# libraries 
import pandas as pd
import numpy as np
import gc

# plotting
import matplotlib.pyplot as plt
import seaborn as sns 

#ml libraries
from sklearn.metrics import confusion_matrix, classification_report, recall_score,precision_score, accuracy_score






def correlation_blockwise(quant_df, ground_truth, data_name):

    print(f"Analysis is starting for {data_name} to get pearson and spearman coefficients")
    print("------"*17)
    
    if ground_truth is None:
        print("Perform pairwise correlation analysis for the whole matrix - no ground truth will be used.")
        quant_df_sub = quant_df.copy()
    else:
        print("Subsetting the matrix protein pairs with those pairs withing the interraction database")
        
        # subset the ground truth with the data 
        proteins_in_pairs = set(ground_truth.iloc[:, 0]).union(set(ground_truth.iloc[:, 1]))
        quant_df_sub = quant_df.loc[quant_df.index.intersection(proteins_in_pairs)]
        print(f"From the protein matrix, in total, {len(proteins_in_pairs)} proteins were found in the database")
        

    # pearson and spearman correlation for protein-protein pairs
    # generate and fix the matrices of each feature type 
    corr_matrix = quant_df_sub.T.corr(method="pearson")
    cor_feat = (corr_matrix
                        .reset_index()
                        .melt(id_vars="index", var_name="Var2", value_name=f"{data_name}_cor_pears")
                        .rename(columns={"index":"Var1"})
                        )
    # remove self correlation and duplicates, sort row wise by strings Protein A > Protein B
    cor_feat = cor_feat[cor_feat["Var1"]!=cor_feat["Var2"]]
    cor_feat = cor_feat[cor_feat["Var1"]>cor_feat["Var2"]]
    pearson_feat = cor_feat

    # same for spearman correlation
    corr_matrix = quant_df_sub.T.corr(method="spearman")
    cor_feat = (corr_matrix
                        .reset_index()
                        .melt(id_vars="index", var_name="Var2", value_name=f"{data_name}_cor_spear")
                        .rename(columns={"index":"Var1"})
                        )
    # remove self correlation and duplicates, sort row wise by strings Protein A > Protein B
    cor_feat = cor_feat[cor_feat["Var1"]!=cor_feat["Var2"]]
    cor_feat = cor_feat[cor_feat["Var1"]>cor_feat["Var2"]]
    spearman_feat = cor_feat
        
    del cor_feat, corr_matrix
    gc.collect()

    # merge 
    result = pd.merge(pearson_feat,spearman_feat, how="left", on=["Var1", "Var2"]) 

    # add ground truth after merging the correlation coefficiens per pair, if ground truth was used
    if ground_truth is not None:
        print(f"Adding ground truth information to protein-protein distance metrics")

        # check membership 
        pair_chars = ground_truth["V1"].astype(str) + "_" + ground_truth["V2"].astype(str)
        result["db"] = np.where((result["Var1"].astype(str) + "_" + result["Var2"].astype(str)).isin(pair_chars), 1, 0) 
            
        # print for sanity check 
        print(f"Posιtive Class: {(result['db'] == 1).sum()} pairs, Negative class: {(result['db'] == 0).sum()} pairs") 

    print(f"The number of pairs generated is {result.shape[0]}")
    print(f"Analysis is completed\n")
    return result 



def compute_distances_blockwise(quant_df_sub,ground_truth=None, data_name="SCBC_raw", 
                               n_chunks = 3, return_dist = "manhattan"):
    """
    A block-based approach to compute pairwise distances in python using numpy and tensor math while
    ignoring NaNs for all proteins in the quant_df_sub protein matrix.

    In the matrix, rows are proteins or peptides, while columns are samples / MS signals 

    Produces a DataFrame with columns [Var1, Var2, std_diff, db], 
    where Var1 > Var2 in string order withour duplicates or self-distances.
    """
    print(f"Analysis is starting for {data_name} for feature {return_dist}")
    print("------"*17)

    # Condition database and distance metric 
    if return_dist not in ["manhattan", "std_difference", "mean_difference", "euclidean", "cosine"]:
        raise RuntimeError("This functions returns either 'manhattan', 'std_differece', 'mean_difference', 'euclidean', or 'cosine' distance metrics for protein pairs")

    if ground_truth is None:
        print("Perform pairwise calculation for the whole matrix - no ground truth will be used.")
    else:
        print("Subsetting the matrix protein pairs with those pairs withing the interraction database")
        proteins_in_pairs = set(ground_truth.iloc[:, 0]).union(set(ground_truth.iloc[:, 1]))
        quant_df_sub = quant_df_sub.loc[quant_df_sub.index.intersection(proteins_in_pairs)]
        print(f"From the protein matrix, in total, {len(proteins_in_pairs)} proteins were found in the database")
        print("\n")
    
    # get suffix for column names
    if return_dist == "std_difference":
        suffix = "std_dif"
    elif return_dist == "manhattan":
        suffix = "man"
    elif return_dist == "mean_difference":
        suffix = "mean_dif"
    elif return_dist == "euclidean":
        suffix = "euc"
    elif return_dist == "cosine":
        suffix = "cos"


    # dims of dataset and division by number of blocks 
    n = quant_df_sub.shape[0]
    chunk_size = n // n_chunks 
    print(f"The size of each block is {chunk_size} rows, and the number of blocks is {n_chunks}")
    print(f"For total number of rows {n}, the number of blocks that are suggested are {n/1100} rounded up.")

    if chunk_size > 1100:
        raise RuntimeError("The size is too large. Increase the n_chunks parameter to reduce size and avoid memory issues")

    # boundaries: [start_of_chunk0, start_of_chunk1,...,start_of_chunkn, end]
    boundaries = [i * chunk_size for i in range(n_chunks)]
    boundaries.append(n)
    all_pairs = []  # list of DataFrames for each block

    for i in range(n_chunks):
        for j in range(i, n_chunks):
            # Subset the rows for chunk i and chunk j
            sub_i = quant_df_sub.iloc[boundaries[i]:boundaries[i+1], :]
            sub_j = quant_df_sub.iloc[boundaries[j]:boundaries[j+1], :]

            mat_i = sub_i.to_numpy()  
            mat_j = sub_j.to_numpy() 

            # Get 
            # shape => (len_i, len_j, num_features)
            # block = mat_i[:, None, :] - mat_j[None, :, :] # moved it inside if statements 

            if return_dist == "std_difference":
                block = mat_i[:, None, :] - mat_j[None, :, :]
                # Compute std of differences across the feature axis (axis=2)
                block_std = np.nanstd(block, axis=2)  # shape => (len_i, len_j)
                del block
                
            elif return_dist == "manhattan": 
                block = mat_i[:, None, :] - mat_j[None, :, :]
                block_mean = np.nansum(np.abs(block), axis=2)
                block_std = block_mean
                del block

            elif return_dist == "mean_difference":
                block = mat_i[:, None, :] - mat_j[None, :, :]
                block_std = np.nanmean(block, axis=2)
                del block

            elif return_dist == "euclidean":
                block = mat_i[:, None, :] - mat_j[None, :, :]
                block_std = np.sqrt(np.nansum(block**2, axis=2))
                del block
            
            elif return_dist == "cosine":
                # Compute the dot products between all pairs in blocks
                dot_products = np.nansum(mat_i[:, None, :] * mat_j[None, :, :], axis=2)
                
                # Compute the L2 norms.
                norm_i = np.sqrt(np.nansum(mat_i**2, axis=1))  # shape (len_i,)
                norm_j = np.sqrt(np.nansum(mat_j**2, axis=1))  # shape (len_j,)
                
                # Create an outer product.
                norms_product = np.outer(norm_i, norm_j)
                
                # Avoid division by zero: replace zeros with np.nan.
                norms_product[norms_product == 0] = np.nan
                
                # Compute cosine similarity and convert to cosine distance.
                cosine_similarity = dot_products / norms_product
                block_std = 1 - cosine_similarity  # cosine distance

            i_labels = sub_i.index.to_numpy()
            j_labels = sub_j.index.to_numpy()

            # Build the pairwise table
            if i == j:
                # Same chunk => we only want the upper triangle 
                # (avoid duplicates, no self-distances)
                tri_i, tri_j = np.triu_indices(len(i_labels), k=1)
                var1 = i_labels[tri_i]
                var2 = i_labels[tri_j]
                dist_vals = block_std[tri_i, tri_j]
            else:
                # Different chunks => all cross pairs
                row_inds, col_inds = np.indices(block_std.shape)
                var1 = i_labels[row_inds.ravel()]
                var2 = j_labels[col_inds.ravel()]
                dist_vals = block_std.ravel()

            # Create a DataFrame for this block
            df_chunk = pd.DataFrame({
                "Var1": var1,
                "Var2": var2,
                f"{data_name}_{suffix}": dist_vals
            })
            all_pairs.append(df_chunk)

            # Clean up memory
            del sub_i, sub_j, mat_i, mat_j, block_std
            gc.collect()

    # Concatenate all partial results
    result = pd.concat(all_pairs, ignore_index=True)

    # Remove any self-distances and duplicates
    result = result[result["Var1"] != result["Var2"]].copy()
    # result = result[result["Var1"]>result["Var2"]] # MAYBE I SHOULD USE THIS AFTER I SORT ALL THE PAIRS BY STRING ORDER VAR1 > VAR 2? 

    # If Var1 < Var2, swap them and keep them by Var1 > Var2
    mask = result["Var1"] < result["Var2"]
    tmp = result.loc[mask, "Var1"]
    result.loc[mask, "Var1"] = result.loc[mask, "Var2"]
    result.loc[mask, "Var2"] = tmp

    if ground_truth is not None:
        print(f"Adding ground truth information to protein-protein distance metrics")

        # check membership 
        pair_chars = ground_truth["V1"].astype(str) + "_" + ground_truth["V2"].astype(str)
        result["db"] = np.where((result["Var1"].astype(str) + "_" + result["Var2"].astype(str)).isin(pair_chars), 1, 0) 
        
        # print for sanity check 
        print(f"Posιtive Class: {(result['db'] == 1).sum()} pairs, Negative class: {(result['db'] == 0).sum()} pairs")

    # A final check-up for self distances and duplicates
    # result = result[result["Var1"]!=result["Var2"]]
    # result = result[result["Var1"]>result["Var2"]]

    print(f"The number of pairs generated is {result.shape[0]}")
    print(f"Analysis is completed")
    print("\n")
    return result



def compute_metrics(y_true, y_predprob, threshold=0.5):
    """
    Given true labels, predicted probabilities, and a threshold,
    computes the confusion matrix and metrics:
      - TPR (sensitivity)
      - FPR
      - FDR = FP / (TP + FP)  (set to np.nan if no predictions)
    Returns:
      conf_mat, TPR, FPR, FDR
    """
    # Predictions: class 1 if probability >= threshold, else 0.
    y_db = np.where(y_predprob > threshold, 1,0)
    
    cm = confusion_matrix(y_true, y_db, labels=[1, 0])
    TP, FN, FP, TN = cm.ravel() 
    
    # Compute metrics; be careful with divisions by zero.
    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    # FDR: proportion of predicted positives that are false.
    FDR = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    
    return cm, TPR, FPR, FDR



def get_probthreshold(y_test, y_predprob, target_FDR, classifier_type, fdr_tol=0.005, tol=0.005, max_iter=50):
    
    """
    Optimizes the classifier probability threshold using a bisection search so that the false discovery rate (FDR) 
    is as close as possible to, but not above, a specified target.

    Notes:
    - The confusion matrix is plotted with the positive class listed first.
    - It implements compute_metrics function.
    - A bisection search is used to find optimal threshold with min probability of 0.5
    - The optimal threshold as well as the confusion matrix are printed.

    """
    
    
    low, high = 0.5, 1.0
    best_threshold = None
    for i in range(max_iter):
        mid = (low + high) / 2
        cm, TPR, FPR, FDR = compute_metrics(y_test, y_predprob, mid)

        print(f"Iteration {i}: threshold={mid:.4f}, FDR={FDR:.4f}, TPR={TPR:.4f}, FPR={FPR:.4f}")
        
        if np.isnan(FDR):
            # No positives predicted; move threshold downward.
            high = mid
        elif FDR - target_FDR > fdr_tol:
            # FDR is too high.
            low = mid
        else:
            best_threshold = mid 
            high = mid
            
        # Check for convergence
        if high - low < tol:
            break

    if best_threshold is None:
        print("Could not find a threshold meeting the target FDR; consider adjusting target_FDR or checking classifier performance.")
    else:
        print(f"\nOptimal threshold found: {best_threshold:.3f}")
        # Compute final confusion matrix and metrics at optimal threshold.
        cm_opt, TPR_opt, FPR_opt, FDR_opt = compute_metrics(y_test, y_predprob, best_threshold)
        print(f"TPR: {TPR_opt:.3f}, FPR: {FPR_opt:.3f}, FDR: {FDR_opt:.3f}")
        
        # --- Plot heatmap of the confusion matrix ---
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d',cmap='YlGnBu',
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'])
    plt.title(f"{classifier_type} Classification Threshold on {round(FDR_opt*100)}% FDR\n p = {best_threshold:.3f}", fontsize=14,
            y=1.05)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()