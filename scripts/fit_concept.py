import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit
import argparse
import warnings
def compute_loss(X, y, n_nonzero_coefs):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False)
    omp.fit(X, y)
    pred = omp.predict(X)
    loss = np.mean((pred - y) ** 2)
    return loss

def find_optimal_n(X, y):
    max_n = X.shape[1]
    low = 640
    high = 670

    # Binary search to find the transition point
    while low < high:
        mid = (low + high) // 2
        if compute_loss(X, y, mid + 1) < compute_loss(X, y, mid):
            low = mid + 1
        else:
            high = mid

    # Check neighboring points to ensure the minimum is found
    candidates = [low]
    if low > 1:
        candidates.append(low - 1)
    if low < max_n:
        candidates.append(low + 1)

    best_n = low
    best_loss = compute_loss(X, y, best_n)
    for n in candidates:
        current_loss = compute_loss(X, y, n)
        if current_loss < best_loss:
            best_loss = current_loss
            best_n = n

    return best_n

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="imagenet")
    parser.add_argument('--concept', default="pcbm")
    parser.add_argument("--n_nonzero_coefs", default=650, type=int)
    args = parser.parse_args()
    data = args.data

    if args.concept == "labo":
        text_emb = np.load(f"file/{data}/lobo_emb.npy")
        with open(f"file/{data}/selected_idx_lobo.pkl", "rb") as f:
            select_idx = pickle.load(f)
    elif args.concept == "pcbm":
        with open(f"file/{data}/selected_concept.pkl", "rb") as f:
            concept_selected = pickle.load(f)
        X = concept_selected[0]
    elif args.concept == "word":
        text_emb = np.load(f"file/words_emb.npy")
        with open(f"file/{data}/selected_idx_word.pkl", "rb") as f:
            select_idx = pickle.load(f)
    print(args)



    img_emb_train = np.load(f"file/{data}/image_emb_train.npy")
    img_emb_train_fit = []
    for i in tqdm(range(len(img_emb_train))):
        y = img_emb_train[i]
        if args.data != "imagenet":
            best_n = find_optimal_n(X, y)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n, fit_intercept=False).fit(X, y)
        else:
            best_n=650
            while(True):
                with warnings.catch_warnings() as w:
                    reg = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n, fit_intercept=False).fit(X, y)
                    if w:
                        best_n -= 5
                    else:
                        break
        img_emb_train_fit.append(reg.predict(X).reshape(1, -1))
    img_emb_train_fit = np.concatenate(img_emb_train_fit, axis=0)
    np.save(f"file/{args.data}/image_emb_train_{args.concept}.npy", img_emb_train_fit)

    if args.data != "imagenet":
        img_emb_val = np.load(f"file/{data}/image_emb_val.npy")
        img_emb_val_fit = []
        for i in tqdm(range(len(img_emb_val))):
            y = img_emb_val[i]
            best_n = find_optimal_n(X, y)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n, fit_intercept=False).fit(X, y)
            img_emb_val_fit.append(reg.predict(X).reshape(1, -1))
        img_emb_val_fit = np.concatenate(img_emb_val_fit, axis=0)
        np.save(f"file/{args.data}/image_emb_val_{args.concept}.npy", img_emb_val_fit)

    img_emb_test = np.load(f"file/{data}/image_emb_test.npy")
    img_emb_test_fit = []
    weights = []
    for i in tqdm(range(len(img_emb_test))):
        y = img_emb_test[i]
        if args.data != "imagenet":
            best_n = find_optimal_n(X, y)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n, fit_intercept=False).fit(X, y)
        else:
            best_n=650
            while(True):
                with warnings.catch_warnings() as w:
                    reg = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n, fit_intercept=False).fit(X, y)
                    if w:
                        best_n -= 5
                    else:
                        break
        img_emb_test_fit.append(reg.predict(X).reshape(1, -1))
        weights.append(reg.coef_.reshape(1, -1))
    img_emb_test_fit = np.concatenate(img_emb_test_fit, axis=0)
    np.save(f"file/{args.data}/image_emb_test_{args.concept}.npy", img_emb_test_fit)
    weights_test = np.concatenate(weights, axis=0)
    np.save(f"file/{args.data}/weights_test_{args.concept}.npy", weights_test)