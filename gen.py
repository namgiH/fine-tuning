# mainly refer to https://colab.research.google.com/github/tripathiaakash/DistilGPT2-Tutorial/blob/main/distilgpt2_fine_tuning.ipynb

from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_tokenizer(weights_dir, device = 'cuda'):
    print("Loading Model ...")
    model = AutoModelForCausalLM.from_pretrained(weights_dir)
    model.to('cuda')
    print("Model Loaded ...")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, padding=True, truncation=True)
    return model, tokenizer

def generate_messages(
    model,
    tokenizer,
    prompt_text,
    stop_token,
    length,
    num_return_sequences,
    temperature = 0.7,
    k=20,
    p=0.9,
    repetition_penalty = 1.0,
    device = 'cuda'
):

    MAX_LENGTH = int(100)
    def adjust_length_to_model(length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length
        
    length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
    return generated_sequences

def run(weights_dir='output', temperature=1.0, k=400, p=0.9, repetition_penalty=1.0, num_return_sequences=5, length=100, stop_token='|EndOfText|', prompt_text='덕기는 안마루에서'):
    model, tokenizer = get_model_tokenizer(weights_dir, device = 'cuda')
    res = generate_messages(
        model,
        tokenizer,
        prompt_text,
        stop_token,
        length,
        num_return_sequences,
        temperature = temperature,
        k=k,
        p=p,
        repetition_penalty = repetition_penalty
    )
    
    print('\n'.join(res))
