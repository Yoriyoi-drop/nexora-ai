use crate::event::Event;
use crossbeam::channel;
use std::collections::VecDeque;

pub struct MessageQueue {
    sender: channel::Sender<Event>,
    receiver: channel::Receiver<Event>,
    replay_buffer: std::sync::Mutex<VecDeque<Event>>,
    max_replay: usize,
}

impl MessageQueue {
    pub fn bounded(cap: usize, replay: usize) -> Self {
        let (tx, rx) = channel::bounded(cap);
        Self {
            sender: tx,
            receiver: rx,
            replay_buffer: std::sync::Mutex::new(VecDeque::with_capacity(replay)),
            max_replay: replay,
        }
    }

    pub fn push(&self, event: Event) -> Result<(), Event> {
        let cloned = event.clone();
        match self.sender.try_send(event) {
            Ok(()) => {
                let mut buf = self.replay_buffer.lock().unwrap();
                if buf.len() >= self.max_replay {
                    buf.pop_front();
                }
                buf.push_back(cloned);
                Ok(())
            }
            Err(channel::TrySendError::Full(e)) => Err(e),
            Err(channel::TrySendError::Disconnected(e)) => Err(e),
        }
    }

    pub fn pop(&self) -> Option<Event> {
        self.receiver.try_recv().ok()
    }

    pub fn pop_blocking(&self) -> Result<Event, channel::RecvError> {
        self.receiver.recv()
    }

    pub fn replay(&self) -> Vec<Event> {
        let buf = self.replay_buffer.lock().unwrap();
        buf.iter().rev().take(self.max_replay).cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    pub fn remaining_capacity(&self) -> Option<usize> {
        self.receiver.capacity().map(|c| c - self.receiver.len())
    }
}
