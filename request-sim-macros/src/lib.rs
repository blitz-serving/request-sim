use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident, Item};

/// Attribute macro to annotate dataset structs and impl blocks with their
/// prompt-text mode (`plain` or `hashed`).
///
/// Both struct and impl items are gated behind the appropriate `#[cfg]`:
///
/// - `#[prompt_text(hashed)]` → `#[cfg(feature = "prompt-text-hashed")]`
/// - `#[prompt_text(plain)]`  → `#[cfg(feature = "prompt-text-plain")]`
#[proc_macro_attribute]
pub fn prompt_text(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mode = parse_macro_input!(attr as Ident);
    let mode_str = mode.to_string();

    let parsed: Item = match syn::parse(item) {
        Ok(item) => item,
        Err(e) => return e.to_compile_error().into(),
    };

    match mode_str.as_str() {
        "hashed" => emit_hashed(parsed),
        "plain" => emit_plain(parsed),
        other => {
            let msg = format!(
                "Invalid prompt_text mode: `{other}`. Expected `plain` or `hashed`."
            );
            quote! { compile_error!(#msg); }.into()
        }
    }
}

/// `#[prompt_text(hashed)]`
///
/// - struct  → gated with `#[cfg(feature = "prompt-text-hashed")]`
/// - impl   → gated with `#[cfg(feature = "prompt-text-hashed")]`
fn emit_hashed(item: Item) -> TokenStream {
    match item {
        Item::Struct(s) => {
            quote! {
                #[cfg(feature = "prompt-text-hashed")]
                #s
            }
            .into()
        }
        Item::Impl(i) => {
            quote! {
                #[cfg(feature = "prompt-text-hashed")]
                #i
            }
            .into()
        }
        _ => {
            quote! {
                compile_error!(
                    "`#[prompt_text]` can only be applied to `struct` or `impl` blocks."
                );
            }
            .into()
        }
    }
}

/// `#[prompt_text(plain)]`
///
/// - struct  → gated with `#[cfg(feature = "prompt-text-plain")]`
/// - impl   → gated with `#[cfg(feature = "prompt-text-plain")]`
fn emit_plain(item: Item) -> TokenStream {
    match item {
        Item::Struct(s) => {
            quote! {
                #[cfg(feature = "prompt-text-plain")]
                #s
            }
            .into()
        }
        Item::Impl(i) => {
            quote! {
                #[cfg(feature = "prompt-text-plain")]
                #i
            }
            .into()
        }
        _ => {
            quote! {
                compile_error!(
                    "`#[prompt_text]` can only be applied to `struct` or `impl` blocks."
                );
            }
            .into()
        }
    }
}
