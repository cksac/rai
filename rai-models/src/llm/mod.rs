pub mod gemma;
pub mod phi2;
pub mod phi3;
pub mod qwen2;

pub mod utils {
    pub struct TokenOutputStream {
        tokenizer: tokenizers::Tokenizer,
        tokens: Vec<u32>,
        current_index: usize,
        prev_index: usize,
    }

    impl TokenOutputStream {
        pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
            Self {
                tokenizer,
                tokens: Vec::new(),
                current_index: 0,
                prev_index: 0,
            }
        }

        pub fn tokens(&self) -> &[u32] {
            &self.tokens
        }

        pub fn len(&self) -> usize {
            self.tokens.len()
        }

        // todo: return Result
        pub fn prompt(&mut self, prompt: impl AsRef<str>) {
            let tokens = self.tokenizer.encode(prompt.as_ref(), true).unwrap();
            self.tokens.extend(tokens.get_ids());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
        }

        // todo: return Result
        fn decode(&self, tokens: &[u32]) -> String {
            self.tokenizer.decode(tokens, true).unwrap()
        }

        // todo: return Result
        pub fn next_token(&mut self, token: u32) -> Option<String> {
            let prev_text = if self.tokens.is_empty() {
                String::new()
            } else {
                let tokens = &self.tokens[self.prev_index..self.current_index];
                self.decode(tokens)
            };

            self.tokens.push(token);
            let text = self.decode(&self.tokens[self.prev_index..]);
            if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
                let ts = text.split_at(prev_text.len());
                self.prev_index = self.current_index;
                self.current_index = self.tokens.len();
                Some(ts.1.to_string())
            } else {
                None
            }
        }
    }
}
