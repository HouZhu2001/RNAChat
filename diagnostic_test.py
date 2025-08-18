#!/usr/bin/env python3

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from rnachat.common.conversation import CONV_VISION, Chat
from rnachat.common.config import Config
from rnachat.common.registry import registry
from rnachat.common.dist_utils import get_rank, init_distributed_mode

# imports modules for registration
from rnachat.datasets.builders import *
from rnachat.models import *
from rnachat.runners import *
from rnachat.tasks import *

def test_conversation_flow():
    """Test 1: Basic conversation flow without model"""
    print("=== Test 1: Conversation Flow ===")
    
    conv = CONV_VISION.copy()
    print(f"System message: {conv.system}")
    print(f"Roles: {conv.roles}")
    print(f"Initial messages: {conv.messages}")
    
    # Add user message
    user_message = "<RNA><RNAHere></RNA> Give me a functional description of this RNA named TEST."
    conv.append_message(conv.roles[0], user_message)
    print(f"After adding user message: {conv.messages}")
    
    # Add assistant placeholder
    conv.append_message(conv.roles[1], None)
    print(f"After adding assistant placeholder: {conv.messages}")
    
    # Generate prompt
    prompt = conv.get_prompt()
    print(f"Generated prompt: {prompt}")
    
    # Test prompt segmentation
    prompt_segs = prompt.split('<RNAHere>')
    print(f"Prompt segments: {prompt_segs}")
    print(f"Number of segments: {len(prompt_segs)}")
    
    print("✅ Conversation flow test passed!\n")
    return True

def test_rna_encoding(model):
    """Test 2: RNA encoding"""
    print("=== Test 2: RNA Encoding ===")
    
    test_seq = "GGCTGGCTTTAGCTCAGCGGTTACTTCGAGTACATTGTAACCACCTCTCTGGGTGGTTCGAGACCCGCGGGTGCTTTCCAGCTCTTTT"
    print(f"Test RNA sequence: {test_seq[:50]}... (length: {len(test_seq)})")
    
    try:
        rna_emb, atts = model.encode_rna([test_seq])
        print(f"RNA embedding shape: {rna_emb.shape}")
        print(f"Attention shape: {atts.shape}")
        print(f"Embedding device: {rna_emb.device}")
        print(f"Embedding dtype: {rna_emb.dtype}")
        
        # Check for NaN or inf values
        if torch.isnan(rna_emb).any():
            print("❌ RNA embeddings contain NaN values!")
            return False
        if torch.isinf(rna_emb).any():
            print("❌ RNA embeddings contain infinite values!")
            return False
            
        print("✅ RNA encoding test passed!\n")
        return True
    except Exception as e:
        print(f"❌ RNA encoding failed: {e}")
        return False

def test_tokenization(model):
    """Test 3: Text tokenization"""
    print("=== Test 3: Text Tokenization ===")
    
    test_text = "Please answer my questions about the following RNA."
    
    try:
        tokens = model.llama_tokenizer(
            test_text, 
            return_tensors="pt", 
            add_special_tokens=True
        )
        print(f"Input text: {test_text}")
        print(f"Token IDs: {tokens.input_ids}")
        print(f"Token shape: {tokens.input_ids.shape}")
        
        # Decode back to text
        decoded = model.llama_tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
        print(f"Decoded text: {decoded}")
        
        print("✅ Tokenization test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False

def test_embedding_generation(model):
    """Test 4: Embedding generation"""
    print("=== Test 4: Embedding Generation ===")
    
    test_text = "Please answer my questions about the following RNA."
    
    try:
        tokens = model.llama_tokenizer(
            test_text, 
            return_tensors="pt", 
            add_special_tokens=True
        ).to(model.llama_model.device)
        
        with torch.no_grad():
            embeddings = model.llama_model.model.model.embed_tokens(tokens.input_ids)
        
        print(f"Text embeddings shape: {embeddings.shape}")
        print(f"Embedding device: {embeddings.device}")
        print(f"Embedding dtype: {embeddings.dtype}")
        
        if torch.isnan(embeddings).any():
            print("❌ Text embeddings contain NaN values!")
            return False
            
        print("✅ Embedding generation test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_context_embedding_integration(chat, model):
    """Test 5: Context embedding integration"""
    print("=== Test 5: Context Embedding Integration ===")
    
    # Create conversation
    conv = CONV_VISION.copy()
    user_message = "<RNA><RNAHere></RNA> Give me a functional description of this RNA named TEST."
    conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)
    
    # Create RNA embeddings
    test_seq = "GGCTGGCTTTAGCTCAGCGGTTACTTCGAGTACATTGTAACCACCTCTCTGGGTGGTTCGAGACCCGCGGGTGCTTTCCAGCTCTTTT"
    rna_emb, _ = model.encode_rna([test_seq])
    img_list = [rna_emb]
    
    try:
        # Get context embeddings
        context_embs = chat.get_context_emb(conv, img_list)
        print(f"Context embeddings shape: {context_embs.shape}")
        print(f"Context device: {context_embs.device}")
        print(f"Context dtype: {context_embs.dtype}")
        
        if torch.isnan(context_embs).any():
            print("❌ Context embeddings contain NaN values!")
            return False
            
        if context_embs.shape[1] == 0:
            print("❌ Context embeddings are empty!")
            return False
            
        print("✅ Context embedding integration test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Context embedding integration failed: {e}")
        return False

def test_simple_generation(model):
    """Test 6: Simple text generation without RNA"""
    print("=== Test 6: Simple Text Generation ===")
    
    test_prompt = "Please provide a brief description of RNA."
    
    try:
        tokens = model.llama_tokenizer(
            test_prompt, 
            return_tensors="pt", 
            add_special_tokens=True
        ).to(model.llama_model.device)
        
        with torch.no_grad():
            outputs = model.llama_model.generate(
                input_ids=tokens.input_ids,
                max_new_tokens=50,
                num_beams=1,
                temperature=0.1,
                do_sample=False,
                pad_token_id=model.llama_tokenizer.eos_token_id
            )
        
        generated_text = model.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input prompt: {test_prompt}")
        print(f"Generated text: {generated_text}")
        
        if len(generated_text.strip()) == 0:
            print("❌ Generated text is empty!")
            return False
            
        if "RNA" not in generated_text and "rna" not in generated_text:
            print("⚠️  Generated text doesn't mention RNA (might be normal)")
            
        print("✅ Simple text generation test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Simple text generation failed: {e}")
        return False

def test_model_generate_method(model):
    """Test 7: Model's generate method"""
    print("=== Test 7: Model Generate Method ===")
    
    test_samples = {
        'seq': ['GGCTGGCTTTAGCTCAGCGGTTACTTCGAGTACATTGTAACCACCTCTCTGGGTGGTTCGAGACCCGCGGGTGCTTTCCAGCTCTTTT'],
        'prompt': ['What is this RNA sequence? <RNAHere> Answer:']
    }
    
    try:
        generated = model.generate(test_samples, max_length=100)
        print(f"Generated: {generated}")
        
        if len(generated) == 0:
            print("❌ Model generate method returned empty result!")
            return False
            
        print("✅ Model generate method test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Model generate method failed: {e}")
        return False

def main():
    print("🔍 RNAChat Diagnostic Test Suite")
    print("=" * 50)
    
    # Initialize model
    print("Initializing model...")
    try:
        # Create args for config
        parser = argparse.ArgumentParser(description="Diagnostic Test")
        parser.add_argument("--cfg-path", help="path to configuration file.",
                            default='configs/rnachat_eval.yaml')
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args([])  # Empty list to use defaults
        
        # Load config
        cfg = Config(args)
        init_distributed_mode(cfg.run_cfg)
        
        # Initialize model
        model_cls = registry.get_model_class(cfg.model_cfg.arch)
        model = model_cls.from_config(cfg.model_cfg).to('cuda:0')
        chat = Chat(model, device='cuda:0')
        
        print("✅ Model initialized successfully!\n")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Run tests
    tests = [
        test_conversation_flow,
        lambda: test_rna_encoding(model),
        lambda: test_tokenization(model),
        lambda: test_embedding_generation(model),
        lambda: test_context_embedding_integration(chat, model),
        lambda: test_simple_generation(model),
        lambda: test_model_generate_method(model)
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The inference pipeline is working correctly.")
        print("If you're still getting poor results, the issue is likely with the model's RNA understanding capabilities.")
    else:
        print("⚠️  Some tests failed. There are technical issues in the inference pipeline.")
        print("Fix the failing tests before blaming the model capabilities.")

if __name__ == "__main__":
    main()
