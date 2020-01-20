import json
import os

from utils_data import load_distinct_data
from utils_gluon import print_infos, setup_bert


# ############################################################################


EXPERIMENT_DIR = "data"


DEFAULT_KWARGS = {
    "gpu": 0,
    "task": "within",
    "description": None,
    #: only for run_name construction, e. g. "traindev", ...
    "prefix": None,
    #: only for run_name construction, e. g. "distinct", ...
    "suffix": None,
    "traindev_split": 0.3,
    "classifier": "pro",
    "is_binary": True,
    "labels": [0, 1],
    "max_seq_len": 128,
    "batch_size": 32,
    "epochs": 3,
    "use_checkpoints": True,
    #: is set dynamically
    "checkpoint_dir": None,
    #: is set dynamically
    "dir": None,
    # "optimizer": "adam",
    # "learning_rate": 5e-6,
    # "epsilon": 1e-9,
    "log_interval": 500,
    #: is set dynamically, currently ignore (epoch_id * datalader_len)
    "global_step": 0,
    "verbose": False,
}


def make_runname(**kwargs):
    prefix = kwargs.get("prefix", None)
    prefix = (str(prefix) + "-") if prefix else ""
    suffix = kwargs.get("suffix", None)
    suffix = ("-" + str(suffix)) if suffix else ""

    task = kwargs.get("task", "within")
    split = kwargs.get("traindev_split", None)
    split = ("_" + str(split)) if split is not None else ""
    seqlen = kwargs.get("max_seq_len", 128)
    bins = "BCE" if kwargs.get("is_binary", True) else "SCE"
    return "{}{}{}_{}_{}{}".format(prefix, task, seqlen, bins, suffix)


def check_rundir(run_name):
    fn_path = os.path.join(EXPERIMENT_DIR, run_name)
    if not os.path.exists(fn_path):
        os.makedirs(fn_path)
    return fn_path


def store_run_params(**kwargs):
    run_name = kwargs["run_name"]
    fn_path = check_rundir(run_name)
    fn = os.path.join(fn_path, "config.json")
    with open(fn, "w", encoding="utf-8") as fp:
        json.dump(kwargs, fp, indent=4)

def load_run_params(**kwargs):
    run_name = kwargs["run_name"]
    fn_path = check_rundir(run_name)
    fn = os.path.join(fn_path, "config.json")
    with open(fn, "r", encoding="utf-8") as fp:
        return json.load(fp)


# ############################################################################


def store_results(path_run, labels, preds):
    fn = os.path.join(path_run, "preds.{}.tsv")
    with open(fn, "w") as fp:
        fp.write("label\tprediction\n")
        for l, p in zip(y_true, y_pred):
            fp.write("{}\t{}\n".format(l, p))


def load_results(path_run):
    fn = os.path.join(path_run, "preds.{}.tsv")
    with open(fn, "r") as fp:
        fp.readline()
        data = list()
        for line in fp:
            label, pred = line.strip().split("\t")
            label, pred = int(label), int(pred)
            data.append(label, pred)
    labels, preds = zip(*data)


# ############################################################################
# distinct datasets


def run_experiment_distinct(**kwargs):
    task = kwargs.get("task", "within")
    num_epochs = kwargs.get("epochs", 3)
    verbose = kwargs.get("verbose", False)
    run_name = kwargs.get("run_name", None)
    if run_name is None:
        run_name = make_runname(**kwargs)
        kwargs["run_name"] = run_name

    path_run = check_rundir(run_name)
    kwargs["dir"] = path_run
    kwargs["checkpoint_dir"] = kwargs["dir"]

    store_run_params(store_run_params)

    with Timer("1 - load {} test/train".format(task)):
        X_train, X_dev, y_train, y_dev = load_distinct_data(task)

    with Timer("2 - setup BERT model"):
        model, vocabulary, ctx, tokenizer, transform, loss_function, metric, all_labels = setup_bert(**kwargs)

    with Timer("3 - prepare training data"):
        data_train_raw, data_train = transform_dataset(X_train, y_train, transform)
        if verbose:
            print_infos(vocabulary, data_train_raw, data_train)

    with Timer("5 - prepare eval data"):
        data_dev_raw, data_dev = transform_dataset(X_dev, y_dev, transform)
        if verbose:
            print_infos(vocabulary, data_dev_raw, data_dev)

    for epoch_id in range(num_epochs):
        with Timer("4 - train model - {}".format(epoch_id)), \
                SummaryWriter(logdir=path_run, flush_secs=60) as sw:
            cur_kwargs = kwargs.copy()
            cur_kwargs["epochs"] = epoch_id + 1
            stats = train(model, data_train, ctx, metric, loss_function, sw=sw, **cur_kwargs)
            plot_train_stats(stats)

        with Timer("6 - evaluate - {}".format(epoch_id)), SummaryWriter(logdir="data/" + run_name, flush_secs=60) as sw:
            all_predictions, cum_loss = predict(model, data_dev, ctx, metric, loss_function, batch_size=6, sw=sw)
            print("Accuracy in epoch {}:".format(epoch_id), metric.get()[1])

            # report results
            y_true, y_pred = predict_out_to_ys(all_predictions, all_labels)
            name="BERTClassifier - distinct {} is_binary={} classifier={}".format(task, kwargs.get("is_binary", True), kwargs.get("classifier", "pro"))
            report_training_results(y_true, y_pred, name=name, heatmap=False)

            # store results
            store_results(path_run, y_true, y_pred)

        # save model again?
        model.save_parameters(os.path.join(path_run, "bert.model.params"))


# ############################################################################

# TODO: is_distinct (data) flag in kwargs?


def run_eval_distinct(store_results=True, **kwargs):
    run_name = kwargs.get("run_name", None)
    if run_name is None:
        run_name = make_runname(**kwargs)
        kwargs["run_name"] = run_name

    old_kwargs = load_run_params(**kwargs)
    path_run = old_kwargs["dir"]

    with Timer("1 - load {} test/train".format(task)):
        X_train, X_dev, y_train, y_dev = load_distinct_data(task)

    with Timer("2 - setup BERT model"):
        model, vocabulary, ctx, tokenizer, transform, loss_function, metric, all_labels = setup_bert(**old_kwargs)

    with Timer("5 - prepare eval data"):
        data_dev_raw, data_dev = transform_dataset(X_dev, y_dev, transform)

    with Timer("6 - evaluate: {}".format(run_name)), SummaryWriter(logdir=kwargs["dir"], flush_secs=60) as sw:
        model.load_parameters(os.path.join(kwargs["dir"], "bert.model.params"), ctx=ctx)
        all_predictions, cum_loss = predict(model, data_dev, ctx, metric, loss_function, sw=sw, **old_kwargs)
        print("Accuracy:", metric.get()[1])

        y_true, y_pred = predict_out_to_ys(all_predictions, all_labels)
        name="BERTClassifier - distinct {} is_binary={} classifier={}".format(task, old_kwargs.get("is_binary", True), old_kwargs.get("classifier", "pro"))
        report_training_results(y_true, y_pred, name=name, heatmap=False)
        
        if store_results:
            store_results(path_run, y_true, y_pred)


# ############################################################################


def print_eval(**kwargs):
    run_name = kwargs.get("run_name", None)
    if run_name is None:
        run_name = make_runname(**kwargs)
        kwargs["run_name"] = run_name

    old_kwargs = load_run_params(**kwargs)

    y_true, y_pred = load_results(old_kwargs["dir"])
    name="BERTClassifier - distinct {} is_binary={} classifier={}".format(task, old_kwargs.get("is_binary", True), old_kwargs.get("classifier", "pro"))
    report_training_results(y_true, y_pred, name=name, heatmap=False)


# ############################################################################



# ############################################################################



# ############################################################################



# ############################################################################



# ############################################################################



# ############################################################################

