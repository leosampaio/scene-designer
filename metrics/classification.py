from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from core.metrics import HistoryMetric
import utils


class PrecomputedValidationAccuracy(HistoryMetric):
    name = 'val-clas-acc'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath = input_data
        pred_labels = pred_y

        return accuracy_score(y, pred_labels)


class PrecomputedTestAccuracy(HistoryMetric):
    name = 'test-clas-acc'
    input_type = 'predictions_on_test_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath = input_data
        pred_labels = pred_y

        return accuracy_score(y, pred_labels)


class SceneGraphAppearenceEmbeddingObjAccuracy(HistoryMetric):
    name = 'embedding-obj-accuracy'
    input_type = 'embedding_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_y, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(y, np.expand_dims(np.argmax(obj_y, axis=1), axis=-1))


class SceneGraphAppearenceEmbeddingSktAccuracy(HistoryMetric):
    name = 'embedding-skt-accuracy'
    input_type = 'embedding_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_y, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(skt_y_gt, np.expand_dims(np.argmax(skt_ys, axis=1), axis=-1))


class SceneGraphObjCropObjAccuracy(HistoryMetric):
    name = 'crops-obj-accuracy'
    input_type = 'crops_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_ys, pred_obj_ys, full_pred_obj_ys, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(y, np.expand_dims(np.argmax(obj_ys, axis=1), axis=-1))


class SceneGraphObjCropSktAccuracy(HistoryMetric):
    name = 'crops-skt-accuracy'
    input_type = 'crops_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_ys, pred_obj_ys, full_pred_obj_ys, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(skt_y_gt, np.expand_dims(np.argmax(skt_ys, axis=1), axis=-1))


class SceneGraphObjCropPredObjAccuracy(HistoryMetric):
    name = 'crops-pred-obj-accuracy'
    input_type = 'crops_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_ys, pred_obj_ys, full_pred_obj_ys, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(y, np.expand_dims(np.argmax(pred_obj_ys, axis=1), axis=-1))


class SceneGraphObjCropFullPredAccuracy(HistoryMetric):
    name = 'crops-full-pred-accuracy'
    input_type = 'crops_classification_predictions_on_validation_set'

    def compute(self, input_data):
        obj_ys, pred_obj_ys, full_pred_obj_ys, skt_ys, y, skt_y_gt = input_data
        return accuracy_score(y, np.expand_dims(np.argmax(full_pred_obj_ys, axis=1), axis=-1))


class SceneGraphAttentionSBIRmAP(HistoryMetric):
    name = 'sbir-mAP'
    input_type = 'SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data = input_data
        return mAP.numpy()


class SceneGraphAttentionSBIRtop1(HistoryMetric):
    name = 'sbir-top1'
    input_type = 'SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data = input_data
        return top1.numpy()


class SceneGraphAttentionSBIRtop5(HistoryMetric):
    name = 'sbir-top5'
    input_type = 'SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data = input_data
        return top5.numpy()


class SceneGraphAttentionSBIRtop10(HistoryMetric):
    name = 'sbir-top10'
    input_type = 'SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data = input_data
        return top10.numpy()


class SceneGraphAttentionObjSBIRmAP(HistoryMetric):
    name = 'obj-sbir-mAP'
    input_type = 'rep_SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return mAP.numpy()


class SceneGraphAttentionObjSBIRtop1(HistoryMetric):
    name = 'obj-sbir-top1'
    input_type = 'rep_SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top1.numpy()


class SceneGraphAttentionObjSBIRtop5(HistoryMetric):
    name = 'obj-sbir-top5'
    input_type = 'rep_SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top5.numpy()


class SceneGraphAttentionObjSBIRtop10(HistoryMetric):
    name = 'obj-sbir-top10'
    input_type = 'rep_SBIR_on_validation_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top10.numpy()


class SceneGraphAttentionSBIRTestmAP(HistoryMetric):
    name = 'sbir-mAP-test'
    input_type = 'SBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return mAP.numpy()


class SceneGraphAttentionSBIRTesttop1(HistoryMetric):
    name = 'sbir-top1-test'
    input_type = 'SBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top1.numpy()


class SceneGraphAttentionSBIRTesttop5(HistoryMetric):
    name = 'sbir-top5-test'
    input_type = 'SBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top5.numpy()


class SceneGraphAttentionSBIRTesttop10(HistoryMetric):
    name = 'sbir-top10-test'
    input_type = 'SBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data
        return top10.numpy()


class SceneGraphAttentionFGSBIRTestmAP(HistoryMetric):
    name = 'fgsbir-mAP-test'
    input_type = 'FGSBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data
        return mAP.numpy()


class SceneGraphAttentionFGSBIRTesttop1(HistoryMetric):
    name = 'fgsbir-top1-test'
    input_type = 'FGSBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data
        return top1.numpy()


class SceneGraphAttentionFGSBIRTesttop5(HistoryMetric):
    name = 'fgsbir-top5-test'
    input_type = 'FGSBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data
        return top5.numpy()


class SceneGraphAttentionFGSBIRTesttop10(HistoryMetric):
    name = 'fgsbir-top10-test'
    input_type = 'FGSBIR_on_test_set'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data
        return top10.numpy()
