use std::collections::HashMap;


/// A simple prefix tree (trie) for detecting shared prefixes across sequences.
/// When `enable_prefix_sharing` is on, sequences with matching prefixes share
/// the initial KV cache computation — the common prefix is processed once
/// and the result is copied to each sequence's KV cache.
#[derive(Debug, Default)]
struct PrefixTrieNode {
    children: HashMap<u32, PrefixTrieNode>,
    /// Sequence IDs that share this prefix
    seq_ids: Vec<u64>,
}

#[derive(Debug)]
pub(crate) struct PrefixTrie {
    root: PrefixTrieNode,
}

impl PrefixTrie {
    pub(crate) fn new() -> Self {
        Self {
            root: PrefixTrieNode::default(),
        }
    }

    /// Insert a sequence's prompt into the trie (full walk).
    pub(crate) fn insert(&mut self, seq_id: u64, prompt: &[u32]) {
        let mut node = &mut self.root;
        for &token in prompt {
            node = node.children.entry(token).or_default();
            node.seq_ids.push(seq_id);
        }
    }

    /// Find the longest prefix of `prompt` shared with another sequence.
    /// Returns `(prefix_len, other_seq_id)` where `other_seq_id` is another
    /// sequence with a matching prefix at that depth. Excludes `exclude_seq_id`.
    /// Returns `None` when no shared prefix of any length exists.
    /// Handles partial matches: if prompt diverges at token P+1, the match
    /// at depth P is still returned.
    pub(crate) fn find_shared_prefix(
        &self,
        prompt: &[u32],
        exclude_seq_id: u64,
    ) -> Option<(usize, u64)> {
        let mut node = &self.root;
        let mut prefix_len = 0usize;
        let mut result: Option<(usize, u64)> = None;
        for &token in prompt {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    prefix_len += 1;
                    if let Some(&other) = node
                        .seq_ids
                        .iter()
                        .find(|&&id| id != exclude_seq_id)
                    {
                        result = Some((prefix_len, other));
                    }
                }
                None => break,
            }
        }
        result
    }
}


