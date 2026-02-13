#![cfg(feature = "alloc")]

use core::{alloc::Layout, cell::Cell, mem::size_of, ptr::NonNull};

#[cfg(feature = "nightly")]
use alloc::{
    alloc::{AllocError, Allocator, Global},
    vec::Vec,
};
#[cfg(not(feature = "nightly"))]
use allocator_api2::{
    alloc::{AllocError, Allocator, Global},
    vec::Vec,
};

use crate::{blink::Blink, local::BlinkAlloc};

#[test]
fn test_local_alloc() {
    let mut blink = BlinkAlloc::new();

    let ptr = blink
        .allocate(Layout::new::<usize>())
        .unwrap()
        .cast::<usize>();
    unsafe {
        core::ptr::write(ptr.as_ptr(), 42);
    }

    blink.reset();
}

#[test]
fn test_bad_iter() {
    struct OneTimeGlobal {
        served: Cell<bool>,
    }

    unsafe impl Allocator for OneTimeGlobal {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if self.served.get() {
                Err(AllocError)
            } else {
                self.served.set(true);
                Global.allocate(layout)
            }
        }

        unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
            Global.deallocate(ptr, layout)
        }
    }

    const ELEMENT_COUNT: usize = 2000;
    const ELEMENT_SIZE: usize = size_of::<u32>();

    let mut blink = Blink::new_in(BlinkAlloc::with_chunk_size_in(
        ELEMENT_SIZE * ELEMENT_COUNT,
        OneTimeGlobal {
            served: Cell::new(false),
        },
    ));

    blink
        .emplace()
        .from_iter((0..ELEMENT_COUNT as u32).filter(|_| true));

    blink.reset();
}

#[test]
fn test_reuse() {
    struct ControlledGlobal {
        enabled: Cell<bool>,
        last: Cell<bool>,
    }

    unsafe impl Allocator for ControlledGlobal {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if !self.enabled.get() {
                return Err(AllocError);
            }
            if self.last.get() {
                self.enabled.set(false);
            }
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            Global.deallocate(ptr, layout)
        }
    }

    let allocator = ControlledGlobal {
        enabled: Cell::new(true),
        last: Cell::new(false),
    };

    let mut alloc = BlinkAlloc::with_chunk_size_in(0, &allocator);

    for _ in 0..123 {
        alloc.allocate(Layout::new::<u32>()).unwrap();
    }
    alloc.reset();

    allocator.last.set(false);

    for _ in 0..123 {
        alloc.allocate(Layout::new::<u32>()).unwrap();
    }
}

#[test]
fn test_emplace_no_drop() {
    use alloc::{borrow::ToOwned, string::String};

    struct Foo<'a>(&'a String);

    impl Drop for Foo<'_> {
        fn drop(&mut self) {
            panic!("Dropped");
        }
    }

    let mut blink = Blink::new();
    let s = "Hello".to_owned();
    let foo = blink.emplace_no_drop().value(Foo(&s));
    assert_eq!(foo.0, "Hello");
    let world = blink.put("World".to_owned());
    // Would be unsound if `foo` could be dropped.
    foo.0 = world;
    blink.reset();
    // assert_eq!(foo.0, "Universe"); // Cannot compile. `foo` does not outlive reset.
}

#[test]
fn test_vec() {
    let mut blink_alloc = BlinkAlloc::new();
    let mut vec = Vec::new_in(&blink_alloc);
    vec.extend([1, 2, 3]);

    vec.push(4);
    vec.extend(5..6);
    vec.push(6);

    assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    drop(vec);
    blink_alloc.reset();
}

#[test]
fn test_tracking_fresh_is_zero() {
    let blink = BlinkAlloc::new();
    assert_eq!(blink.allocated_bytes(), 0);
    assert_eq!(blink.total_capacity(), 0);
}

#[test]
fn test_tracking_single_alloc() {
    let blink = BlinkAlloc::new();
    blink.allocate(Layout::new::<u64>()).unwrap();
    assert!(blink.allocated_bytes() >= 8);
    assert!(blink.total_capacity() >= blink.allocated_bytes());
}

#[test]
fn test_tracking_monotonic_increase() {
    let blink = BlinkAlloc::new();
    let layout = Layout::new::<u32>();
    let mut prev = 0;
    for _ in 0..100 {
        blink.allocate(layout).unwrap();
        let current = blink.allocated_bytes();
        assert!(current >= prev + 4,
            "allocated_bytes should grow by at least 4: prev={}, current={}",
            prev, current);
        prev = current;
    }
}

#[test]
fn test_tracking_reset_zeros_bytes() {
    let mut blink = BlinkAlloc::new();
    for _ in 0..50 {
        blink.allocate(Layout::new::<[u8; 64]>()).unwrap();
    }
    assert!(blink.allocated_bytes() > 0);
    blink.reset();
    assert_eq!(blink.allocated_bytes(), 0);
    assert!(blink.total_capacity() > 0, "last chunk should be retained");
}

#[test]
fn test_tracking_reset_final_zeros_everything() {
    let mut blink = BlinkAlloc::new();
    for _ in 0..50 {
        blink.allocate(Layout::new::<[u8; 64]>()).unwrap();
    }
    blink.reset_final();
    assert_eq!(blink.allocated_bytes(), 0);
    assert_eq!(blink.total_capacity(), 0);
}

#[test]
fn test_tracking_warmup_exact_u64() {
    let mut blink = BlinkAlloc::new();
    let layout = Layout::new::<u64>();
    for _ in 0..200 {
        blink.allocate(layout).unwrap();
    }
    blink.reset();

    // Single chunk after warm-up. ChunkHeader base is aligned to
    // align_of::<ChunkHeader>() which is >= 8, so u64 allocs pack
    // with no padding.
    for _ in 0..200 {
        blink.allocate(layout).unwrap();
    }
    assert_eq!(blink.allocated_bytes(), 200 * 8);
}

#[test]
fn test_tracking_warmup_exact_u32() {
    let mut blink = BlinkAlloc::new();
    let layout = Layout::new::<u32>();
    for _ in 0..500 {
        blink.allocate(layout).unwrap();
    }
    blink.reset();

    for _ in 0..500 {
        blink.allocate(layout).unwrap();
    }
    assert_eq!(blink.allocated_bytes(), 500 * 4);
}

#[test]
fn test_tracking_warmup_exact_mixed() {
    let mut blink = BlinkAlloc::new();
    // Allocate a pattern that includes alignment padding.
    let l1 = Layout::new::<u8>();  // size 1, align 1
    let l2 = Layout::new::<u64>(); // size 8, align 8
    for _ in 0..100 {
        blink.allocate(l1).unwrap();
        blink.allocate(l2).unwrap();
    }
    let first_pass = blink.allocated_bytes();
    blink.reset();

    // Second pass: same pattern, same single chunk, should be identical.
    for _ in 0..100 {
        blink.allocate(l1).unwrap();
        blink.allocate(l2).unwrap();
    }
    assert_eq!(blink.allocated_bytes(), first_pass);
    // Each pair: 1 byte for u8, then up to 7 bytes padding, then 8 bytes
    // for u64 = 16 bytes per pair. Cursor advances by 16 each time.
    assert_eq!(blink.allocated_bytes(), 100 * 16);
}

#[test]
fn test_tracking_capacity_stabilizes() {
    let mut blink = BlinkAlloc::new();
    let layout = Layout::new::<u32>();
    let mut prev_cap = 0;
    for cycle in 0..10 {
        for _ in 0..500 {
            blink.allocate(layout).unwrap();
        }
        let cap = blink.total_capacity();
        blink.reset();
        if cycle >= 2 {
            assert_eq!(cap, prev_cap,
                "capacity should stabilize: cycle {}", cycle);
        }
        prev_cap = cap;
    }
}

#[test]
fn test_tracking_aligned_allocs() {
    let blink = BlinkAlloc::new();
    blink.allocate(Layout::from_size_align(1, 1).unwrap()).unwrap();
    let a1 = blink.allocated_bytes();
    blink.allocate(Layout::from_size_align(1, 16).unwrap()).unwrap();
    let a2 = blink.allocated_bytes();
    blink.allocate(Layout::from_size_align(1, 64).unwrap()).unwrap();
    let a3 = blink.allocated_bytes();
    assert!(a1 > 0);
    assert!(a2 > a1);
    assert!(a3 > a2);
}

#[test]
fn test_tracking_multi_chunk() {
    let blink = BlinkAlloc::new();
    let small = Layout::from_size_align(8, 8).unwrap();
    let big = Layout::from_size_align(4096, 8).unwrap();
    for _ in 0..30 {
        blink.allocate(small).unwrap();
    }
    let before = blink.allocated_bytes();
    blink.allocate(big).unwrap();
    assert!(blink.allocated_bytes() >= before + 4096);
}

#[test]
fn test_tracking_approximation_error() {
    let mut blink = BlinkAlloc::new();
    let layout = Layout::new::<u32>();
    let count = 10_000;
    for _ in 0..count {
        blink.allocate(layout).unwrap();
    }
    let exact_data = count * 4;
    let reported = blink.allocated_bytes();
    assert!(reported >= exact_data);
    let overhead = reported - exact_data;
    assert!(overhead < exact_data / 10,
        "overhead ({}) should be < 10% of data ({})", overhead, exact_data);

    // After warm-up the error should be near zero.
    blink.reset();
    for _ in 0..count {
        blink.allocate(layout).unwrap();
    }
    let warm = blink.allocated_bytes();
    let warm_overhead = warm - exact_data;
    assert!(warm_overhead < exact_data / 100,
        "warm overhead ({}) should be < 1% of data ({})", warm_overhead, exact_data);
}

#[test]
fn test_tracking_vec() {
    let mut blink = BlinkAlloc::new();
    let mut v: Vec<u64, _> = Vec::new_in(&blink);
    v.extend(0u64..100);
    drop(v);
    assert!(blink.allocated_bytes() > 0);
    blink.reset();
    assert_eq!(blink.allocated_bytes(), 0);
}

#[test]
fn test_tracking_capacity_gte_allocated() {
    let blink = BlinkAlloc::new();
    let layouts = [
        Layout::new::<u8>(),
        Layout::new::<u32>(),
        Layout::new::<u64>(),
        Layout::new::<[u8; 128]>(),
        Layout::new::<[u64; 32]>(),
    ];
    for layout in layouts.iter().cycle().take(500) {
        blink.allocate(*layout).unwrap();
        assert!(blink.total_capacity() >= blink.allocated_bytes());
    }
}

#[test]
fn test_tracking_capacity_monotonic() {
    let blink = BlinkAlloc::new();
    let layout = Layout::new::<u64>();
    let mut prev = 0;
    for _ in 0..500 {
        blink.allocate(layout).unwrap();
        let cap = blink.total_capacity();
        assert!(cap >= prev);
        prev = cap;
    }
}

#[test]
fn test_tracking_custom_chunk_size() {
    let small = BlinkAlloc::with_chunk_size(64);
    let large = BlinkAlloc::with_chunk_size(8192);
    small.allocate(Layout::new::<u8>()).unwrap();
    large.allocate(Layout::new::<u8>()).unwrap();
    assert!(large.total_capacity() > small.total_capacity());
}

#[test]
fn test_tracking_zero_size_layout() {
    let blink = BlinkAlloc::new();
    blink.allocate(Layout::from_size_align(0, 1).unwrap()).unwrap();
    let _ = blink.allocated_bytes();
    let _ = blink.total_capacity();
}

#[test]
fn test_tracking_huge_alloc() {
    let blink = BlinkAlloc::new();
    let size = 1024 * 1024;
    blink.allocate(Layout::from_size_align(size, 8).unwrap()).unwrap();
    assert!(blink.allocated_bytes() >= size);
    assert!(blink.total_capacity() >= size);
}

#[test]
fn test_tracking_blink_put() {
    let mut blink = Blink::new();
    assert_eq!(blink.allocator().allocated_bytes(), 0);
    blink.put(42u64);
    assert!(blink.allocator().allocated_bytes() > 0);
    blink.put(123u32);
    assert!(blink.allocator().allocated_bytes() > 0);
    blink.reset();
    assert_eq!(blink.allocator().allocated_bytes(), 0);
}

#[test]
fn test_tracking_blink_emplace_iter() {
    let mut blink = Blink::new();
    let _slice = blink.emplace().from_iter(0u64..50);
    assert!(blink.allocator().allocated_bytes() >= 400,
        "should be >= 400 (50 u64s), got {}", blink.allocator().allocated_bytes());
    blink.reset();
    assert_eq!(blink.allocator().allocated_bytes(), 0);
}

#[test]
fn test_tracking_blink_copy_str() {
    let blink = Blink::new();
    let s = "hello world, this is a test string!";
    blink.copy_str(s);
    assert!(blink.allocator().allocated_bytes() >= s.len());
}
