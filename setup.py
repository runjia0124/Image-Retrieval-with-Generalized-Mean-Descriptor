from utils.retrieval_feature import AntiFraudFeatureDataset
import pickle
import cv2

if __name__ == '__main__':

    hash_size = 0
    input_dim = 2048
    num_hashtables = 1
    img_dir = '.\\database\\data'
    test_img_dir = '.\\images'
    network = '.\\weights\\gl18-tl-resnet50-gem-w-83fdc30.pth'

    # feature_dict, lsh = AntiFraudFeatureDataset(img_dir, network).constructfeature(hash_size, input_dim, num_hashtables)

    # with open("./feature", "rb") as f:
    #     feature_dict = pickle.load(f)
    with open("./index", "rb") as f:
        lsh = pickle.load(f)
    test_feature_dict = AntiFraudFeatureDataset(test_img_dir, network).test_feature()
    num_results = 4
    cnt = 1

    for q_path, q_vec in test_feature_dict.items():
        response = lsh.query(q_vec.flatten(), num_results=int(num_results), distance_func="cosine")

        for i in range(num_results):
            for item in response[i][0][1]:
                img = cv2.imread(item)
                cv2.imshow("query"+str(cnt)+"_result"+str(i+1), img)
                cv2.imwrite("query"+str(cnt)+"_result"+str(i+1)+item[-4:], img)
            cv2.waitKey(0)
        cnt += 1
