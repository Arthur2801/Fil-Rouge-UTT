from dealabs import Dealabs
import json

def main():
    client = Dealabs()
    
    # Récupérer les 1000 derniers deals
    all_deals = []
    page = 0
    per_page = 50  # Limite max de l'API
    nb_deals_to_fetch = 10
    
    while len(all_deals) < nb_deals_to_fetch:
        response = client.get_new_deals(params={'page': page, 'limit': per_page})
        deals_data = response.get('data', [])
        
        if not deals_data:
            break
            
        all_deals.extend(deals_data)
        page += 1
        print(f"Récupéré {len(all_deals)} deals...")
    
    all_deals = all_deals[:nb_deals_to_fetch]  
    deals_by_thread_id = {}
    for i, deal in enumerate(all_deals):
        thread_id = deal.get('thread_id') or deal.get('id')
        comments = client.get_thread_comments(thread_id)
        deal['comments'] = [c.to_dict() for c in comments]
        deals_by_thread_id[str(thread_id)] = deal

        #print(f"[{i+1}/{len(all_deals)}] Deal {thread_id}: {len(comments)} commentaires")
    
    # Afficher les résultats
    with open("deals_with_comments.json", "w", encoding="utf-8") as f:
        json.dump(deals_by_thread_id, f, ensure_ascii=False, indent=2)

    
    print(f"\n✅ Export terminé: deals_with_comments.json ({len(all_deals)} deals)")


if __name__ == "__main__":
    main()