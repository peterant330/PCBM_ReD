import os
import argparse
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import LogisticRegression
import random
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept', default="pcbm")
    parser.add_argument('--data', default="imagenet")
    parser.add_argument('--intercept', action='store_true')

    args = parser.parse_args()
    data = args.data
    intercept = args.intercept


    img_emb_train_fit = np.load(f"file/{data}/image_emb_train_{args.concept}.npy")
    # img_emb_train /= np.linalg.norm(img_emb_train, axis=1, keepdims=True)
    img_emb_val_fit = np.load(f"file/{data}/image_emb_val_{args.concept}.npy")
    # img_emb_val /= np.linalg.norm(img_emb_val, axis=1, keepdims=True)
    img_emb_test_fit = np.load(f"file/{data}/image_emb_test_{args.concept}.npy")

    with open(f"file/{data}/target.pkl", "rb") as f:
        train_label, val_label, test_label = pickle.load(f)
    img_emb_val_fit, val_label = img_emb_test_fit, test_label

    val_acc_step_list = np.zeros([3, 8])
    best_c_weights_list = []
    train_features, val_features, test_features = img_emb_train_fit, img_emb_val_fit, img_emb_test_fit
    train_labels, val_labels, test_labels = train_label, val_label, test_label
    
    for seed in range(1, 4):
        np.random.seed(seed)
        random.seed(seed)
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight, fit_intercept=intercept).fit(train_features,
                                                                                                  train_labels)
            pred = clf.predict(val_features)
            acc_val = np.mean([int(t == p) for t, p in zip(val_labels, pred)]).astype(np.float32) * 100.
            acc_list.append(acc_val)

        print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak


        def binary_search(c_left, c_right, seed, step, val_acc_step_list):
            clf_left = LogisticRegression(  # random_state=0,
                C=c_left,
                max_iter=1000,
                verbose=0,
                n_jobs=4, fit_intercept=intercept)
            clf_left.fit(train_features, train_labels)
            pred_left = clf_left.predict(val_features)
            accuracy = np.mean((val_labels == pred_left).astype(np.float32)) * 100.
            acc_left = np.mean([int(t == p) for t, p in zip(val_labels, pred_left)]).astype(np.float32) * 100
            print("Val accuracy (Left): {:.2f}".format(acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right, fit_intercept=intercept).fit(train_features,
                                                                                                       train_labels)
            pred_right = clf_right.predict(val_features)
            acc_right = np.mean([int(t == p) for t, p in zip(val_labels, pred_right)]).astype(np.float32) * 100
            print("Val accuracy (Right): {:.2f}".format(acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(val_features)
            val_acc = np.mean([int(t == p) for t, p in zip(val_labels, pred)]).astype(np.float32) * 100
            print("Val Accuracy: {:.2f}".format(val_acc), flush=True)
            val_acc_step_list[seed - 1, step] = val_acc

            return (
                np.power(10, c_left),
                np.power(10, c_right),
                seed,
                step,
                val_acc_step_list,
            )


        for step in range(8):
            c_left, c_right, seed, step, val_acc_step_list = binary_search(c_left, c_right, seed, step, val_acc_step_list)

        # save c_left as the optimal weight for each run
        best_c_weights_list.append(c_left)


    best_c = np.mean(best_c_weights_list)
    classifier = LogisticRegression(random_state=0,
                                    C=best_c,
                                    max_iter=1000,
                                    verbose=0, fit_intercept=intercept)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    # test performance
    accuracy = np.mean((test_labels == predictions).astype(np.float32)) * 100.
    print(args)
    print(best_c)
    print(f"Test Accuracy = {accuracy:.3f}")
