from rewards import verify_reward_func, ai_feedback_reward_func

def test_rewards():
    print("Testing Reward Functions...")
    
    # Test 1: Verification
    # Case A: Correct
    ans_correct = ["Reasoning... \\boxed{42}"]
    truth = ["42"]
    r = verify_reward_func(ans_correct, answer=truth)
    print(f"Test 1A (Correct): {r} -> Expect [1.0]")
    assert r[0] == 1.0, "Verification failed on correct answer"

    # Case B: Incorrect
    ans_wrong = ["Reasoning... \\boxed{0}"]
    r = verify_reward_func(ans_wrong, answer=truth)
    print(f"Test 1B (Incorrect): {r} -> Expect [0.0]")
    assert r[0] == 0.0, "Verification failed on incorrect answer"

    # Test 2: AI Feedback (Heuristic)
    # Case A: Good structure
    good_text = ["Step 1: Do this. Therefore, the answer is \\boxed{10}"]
    r = ai_feedback_reward_func(good_text)
    print(f"Test 2A (Good): {r} -> Expect > 0.4")
    assert r[0] >= 0.4, "AI Feedback failed on good text"

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_rewards()
