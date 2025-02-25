import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv
from utils import generate_confusion_matrix, plot_confusion_matrix


def get_first_pixel_colors(folder_path, teams):
    team_color = []
    for file_name in teams:

        image = cv2.imread(os.path.join(folder_path, file_name + '.jpg'))
        first_pixel_color = image[0, 0]
        team_color.append(first_pixel_color.tolist())
    return team_color


def filter_colors(clustered_colors, team_colors, threshold=80):
    filtered_colors_raw = []
    filtered_colors = []
    indices = []
    similarities = []

    for index, color1 in enumerate(clustered_colors):
        min_similarity = float('inf')
        closest_color = None

        for color2 in team_colors:
            similarity = np.linalg.norm(np.array(color1) - np.array(color2))
            if similarity < min_similarity:
                min_similarity = similarity
                closest_color = color2

        if min_similarity < threshold:
            filtered_colors_raw.append([int(c) for c in color1.tolist()])
            filtered_colors.append(closest_color)
            similarities.append(min_similarity)
            indices.append(index)

    return filtered_colors_raw, filtered_colors, indices, similarities


def process_images(folder_path, team_colors, output_folder, teams):
    # dominant_colors = []
    outputs = []
    team = folder_path.split('/')[-1]
    os.makedirs(os.path.join(output_folder, team), exist_ok=True)
    # output_path = os.path.join(output_folder, os.path.basename(folder_path))
    for file_name in os.listdir(folder_path):
        output_path = os.path.join(output_folder, os.path.basename(folder_path), file_name)
        os.makedirs(os.path.join(output_folder, os.path.basename(folder_path)), exist_ok=True)
        if file_name.startswith('.'):
            continue
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        image_2d = image.reshape(-1, 3)

        # team_colors = get_first_pixel_colors(team_colors_folder)
        initial_centers1 = np.array(team_colors)
        kmeans1 = KMeans(n_clusters=len(initial_centers1), init=initial_centers1, n_init=1)
        kmeans1.fit(image_2d)
        cluster_centers = kmeans1.cluster_centers_  # 聚类颜色
        filtered_colors_raw, filtered_colors, indices, similarity = filter_colors(cluster_centers, team_colors,
                                                                                  80)  # caculate_distance,min_distance
        labels_1 = kmeans1.predict(image_2d)
        label_counts_1 = np.bincount(labels_1)
        total_samples = len(labels_1)
        ratios_1 = label_counts_1 / total_samples

        index_list = []
        labels = kmeans1.labels_  # pixel2
        masks = np.zeros_like(labels)
        for index, label in enumerate(labels):
            if label in indices:
                masks[index] = 255
                index_list.append(index)
        pixels = [image_2d[idx] for idx in index_list]
        masks = masks.reshape(image.shape[0], image.shape[1], )
        # Add mask to 3d
        masks = np.stack([masks] * 3, axis=-1)
        # convert to uint
        masks = masks.astype(np.uint8)

        # cv2.imshow("mask", masks)
        # cv2.waitKey(0)

        filtered_team_color = list(set([tuple(color) for color in filtered_colors]))
        initial_centers2 = np.array(filtered_team_color)
        kmeans2 = KMeans(n_clusters=len(initial_centers2), init=initial_centers2, n_init=1)
        kmeans2.fit(pixels)
        color2 = kmeans2.cluster_centers_
        filtered_colors_raw2, filtered_colors2, indices2, similarity2 = filter_colors(color2, initial_centers2,
                                                                                      99999)

        labels_2 = kmeans2.predict(pixels)
        label_counts_2 = np.bincount(labels_2)
        total_samples2 = len(labels_2)
        ratios_2 = label_counts_2 / total_samples2

        for idx, color2 in enumerate(filtered_colors2):
            if similarity2[idx] == min(similarity2):
                final_color = color2


        team_idx = team_colors.index(final_color.tolist())

        first_img = np.zeros((500, 700, 3), np.uint8)
        resized_img = cv2.resize(image, (500, 500))
        resized_mask = cv2.resize(masks, (500, 500))
        first_img[0:500, 0:500] = resized_img
        for i, color in enumerate(filtered_colors_raw):
            first_img[i * 100:(i + 1) * 100, 500:600] = color
            first_img[i * 100:(i + 1) * 100, 600:700] = filtered_colors[i]
            cv2.putText(first_img, 'sim:{}'.format(int(similarity[i])), (550, 100 * i + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(first_img, 'ratio:{}'.format(round(ratios_1[i], 2)), (550, 100 * i + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1, cv2.LINE_AA)

        second_img = np.zeros((500, 200, 3), np.uint8)
        for i, color in enumerate(filtered_colors_raw2):
            second_img[i * 100:(i + 1) * 100, 0:100] = color
            second_img[i * 100:(i + 1) * 100, 100:200] = filtered_colors2[i]
            second_img[400:500, 100:200] = final_color
            cv2.putText(second_img, 'sim:{}'.format(int(similarity2[i])), (750, 100 * i + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(second_img, 'ratio:{}'.format(round(ratios_2[i], 2)), (750, 100 * i + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1, cv2.LINE_AA)

        full_img = np.concatenate([first_img, resized_mask, second_img], axis=1)
        cv2.imwrite(output_path, full_img)
        # cv2.imwrite(os.path.join(output_path, 'second_cluster.jpg'), new_img)
        # cv2.imwrite(os.path.join(output_path, 'raw.jpg'), image)
        outputs.append([file_name, team, teams[team_idx]])
    return outputs


def process_game(root_folder, output_root, game_folder, log_file):
    output_folder = os.path.join(output_root, game_folder)
    checker_folder = os.path.join(root_folder, game_folder, 'check')
    teams = ["player1", "player2", "goalkeeper1", "goalkeeper2", "referee"]
    ref_folder = os.path.join(root_folder, game_folder, 'ref')
    team_colors = get_first_pixel_colors(ref_folder, teams)
    output_csv = os.path.join(output_folder, 'results.csv')

    outputs = [["raw_img", "class", "results"]]
    all_cnt, all_correct = 0, 0
    accuracy = []
    with open(log_file, 'a') as f:
        f.write(f"Processing Game: {game_folder}\n")
    for team in teams:
        cnt, correct = 0, 0
        team_out = process_images(os.path.join(checker_folder, team), team_colors, output_folder, teams)
        outputs += team_out
        cnt += len(team_out)
        all_cnt += len(team_out)
        for out in team_out:
            if os.path.basename(out[1]) == out[2]:
                correct += 1
                all_correct += 1

        accuracy.append(correct / cnt)
        print(f"Game: {game_folder}, Team: {team}, Total: {cnt}, Correct: {correct}, Accuracy: {correct / cnt}")
        with open(log_file, 'a') as f:
            f.write(f"Game: {game_folder}, Team: {team}, Total: {cnt}, Correct: {correct}, Accuracy: {correct / cnt}\n")

    # print(outputs)
    game_preds = [[o[1], o[2]] for o in outputs[1:]]
    confusion_matrix = generate_confusion_matrix(game_preds)
    plot_confusion_matrix(confusion_matrix, save_path=os.path.join(output_folder, 'confusion_matrix.png'))

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        for line in outputs:
            writer.writerow(line)

    print(f"Total: {all_cnt}, Correct: {all_correct}")
    print(f"Accuracy: {all_correct / all_cnt}")


if __name__ == '__main__':
    root_folder = "knn_assets/game1_outer"
    output_root = "knn_assets/out2"
    os.makedirs(output_root, exist_ok=True)
    log_file = os.path.join(output_root, 'log.txt')

    games_folder = os.listdir(root_folder)
    for game_folder in games_folder:
        if game_folder.startswith('.'):
            continue
        process_game(root_folder, output_root, game_folder, log_file)
        # writer.writerows(outputs)
    # process_images('knn_assets/team_colors/raw_img/yellow', team_colors)

    # process_game(root_folder, output_root, 'game1', log_file)

