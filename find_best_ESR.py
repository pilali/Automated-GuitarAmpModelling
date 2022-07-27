import json
from pathlib import Path

maindict = {}

result = list(Path(".").rglob("training_stats.json"))

for f in result:
    with open(f) as json_file:
        data = json.load(json_file)
        test_lossESR_final = data['test_lossESR_final']
        test_lossESR_best = data['test_lossESR_best']
        tmp = min(test_lossESR_final, test_lossESR_best)
        path = f.parents[0]
        if tmp == test_lossESR_final:
            model = path / "model.json"
        else:
            model = path / "model_best.json"
        valuedict = {}
        valuedict['file'] = model
        with open(model) as model_json:
            model_data = json.load(model_json)
            valuedict['model_data'] = model_data['model_data']
        valuedict['epoch'] = data['current_epoch']
        valuedict['best_val_loss'] = data['best_val_loss']
        valuedict['test_lossESR_final'] = test_lossESR_final
        valuedict['test_lossESR_best'] = test_lossESR_best
        maindict[tmp] = valuedict

orderedlist = sorted(maindict)

counter = 1
for item in orderedlist:
    print("%d) %s: %s %d skip = %d" % (counter, maindict[item]['model_data']['model'], maindict[item]['model_data']['unit_type'], maindict[item]['model_data']['hidden_size'], maindict[item]['model_data']['skip']))
    print("    training epochs: %d" % maindict[item]['epoch'])
    print("    best_val_loss: %.8f" % maindict[item]['best_val_loss'])
    print("    best ESR: %.8f" % item)
    print("    test_lossESR_final = %.8f test_lossESR_best = %.8f" % (maindict[item]['test_lossESR_final'], maindict[item]['test_lossESR_best']))
    print("    file: %s" % maindict[item]['file'])
    counter = counter + 1
