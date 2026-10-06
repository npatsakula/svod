use svod_ir::{ConstValue, DType, UOp};

#[test]
fn test_tree_simple() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let sum = a.try_add(&b).unwrap();

    let tree = sum.tree();
    println!("Tree output:\n{}", tree);
    assert!(tree.contains("Add"));
    assert!(tree.contains("CONST"));
}

#[test]
fn test_tree_shared_nodes() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let shared = a.try_add(&a).unwrap();

    // Compact tree should show back-reference
    let compact = shared.tree();
    println!("Compact tree:\n{}", compact);
    assert!(compact.contains("see above"));

    // Full tree should NOT show back-reference
    let full = shared.tree_full();
    println!("Full tree:\n{}", full);
    assert!(!full.contains("see above"));
}

/// The branch glyphs only, with each line's node label cut off.
fn skeleton(tree: &str) -> Vec<&str> {
    tree.lines().map(|line| &line[..line.find('[').expect("every line names a node")]).collect()
}

/// `(a + b) * c` with `a + b` shared: the last child turns the corner, an
/// earlier sibling's subtree keeps the vertical rule, and a repeated node is
/// printed once in the compact tree.
#[test]
fn tree_draws_branches_and_back_references() {
    let [a, b, c] = [1.0, 2.0, 3.0].map(|v| UOp::const_(DType::Float32, ConstValue::Float(v)));
    let sum = a.try_add(&b).unwrap();
    let root = sum.try_mul(&c).unwrap().try_add(&sum).unwrap();

    let full = root.tree_full();
    assert_eq!(
        skeleton(&full),
        ["", "├── ", "│   ├── ", "│   │   ├── ", "│   │   └── ", "│   └── ", "└── ", "    ├── ", "    └── "],
        "{full}"
    );

    let compact = root.tree();
    assert_eq!(
        skeleton(&compact),
        ["", "├── ", "│   ├── ", "│   │   ├── ", "│   │   └── ", "│   └── ", "└── "],
        "{compact}"
    );
    assert_eq!(compact.lines().last().unwrap(), format!("└── [{}] → (see above)", sum.id), "{compact}");
}
