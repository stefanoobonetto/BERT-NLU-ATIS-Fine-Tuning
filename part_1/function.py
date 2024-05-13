import torch 
from conll import evaluate
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def create_next_test_folder(base_dir):
    global n, dir  # Declare as global to update the global variables
    existing_folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    last_number = 0
    for folder in existing_folders:
        if folder.startswith("test_"):
            try:
                number = int(folder.split("_")[1])
                last_number = max(last_number, number)
            except ValueError:
                pass

    n = last_number

    next_folder = os.path.join(base_dir, f"test_{last_number + 1}")
    os.makedirs(next_folder, exist_ok=True)
    dir = next_folder
    print(f"Created folder: {next_folder}")
    return next_folder

def plot_line_graph(sampled_epochs, losses_dev, losses_train, filename):
    y1 = losses_dev  # l'ultimo valore è il best_ppl, non voglio plottarlo 
    y2 = losses_train

    x1 = list(range(1, len(y1) + 1))  # Indici incrementati di 1
    x2 = list(range(1, len(y2) + 1))  # Indici incrementati di 1
    

    plt.plot(x1, y1, label='Validation Loss')
    plt.plot(x2, y2, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses for each epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Loss'])  # Scrivi l'intestazione delle colonne
        for idx, value in enumerate(data):
            writer.writerow([idx + 1, value])

def save_results(lr, epoch, sampled_epochs, losses_dev, losses_train, drop, bidir):
    
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(dir, "results")
    dir = create_next_test_folder(dir)

    test = "[LSTM_"
    if drop:
        test += "drop_"
    if bidir:
        test += "bidirectional_"
    plot_line_graph(sampled_epochs, losses_dev, losses_train, os.path.join(dir, "loss_" + test + ".png"))
    save_to_csv(losses_dev, os.path.join(dir, "ppls_dev_" + test + ".csv"))
    save_to_csv(losses_train, os.path.join(dir, "" + test + ".csv"))

    print("Experiment stopped at epoch: ", epoch, " with lr: ", lr, "[drop: ", drop, ", bidirectional: ", bidir, "]")

